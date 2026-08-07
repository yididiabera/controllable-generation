import torch
import torch.nn as nn
from pathlib import Path
import sys
import time


import torch.nn.functional as F
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
WAN_PATH = project_root / 'Wan2.2'

if not WAN_PATH.exists():
    raise RuntimeError(f"WAN directory not found at {WAN_PATH}")

sys.path.insert(0, str(WAN_PATH))

from wan.modules import WanModel, Wan2_2_VAE, T5EncoderModel
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.configs import WAN_CONFIGS

sys.path.insert(0, str(current_file.parent.parent))
from depth_models.control_adapter import ControlAdapter


class ZeroLinear(nn.Module):
    

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)



class ControllableWAN(nn.Module):

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = 'cuda',
        control_injection_layers: list = [0,4,8,12,16,20,24,28],
        spatial_downsample: int = 16,
    ):
        super().__init__()

        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.control_injection_layers = control_injection_layers

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        print(f"\n{'='*70}")
        print("Loading Controllable WAN 2.2  (ControlNet zero-conv injection)")
        print(f"{'='*70}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")

        print("  [1/5] Loading VAE...")
        self.vae = self._load_vae()

        print("  [2/5] Loading WAN DiT...")
        self.wan = self._load_wan()

        print("  [3/5] Loading T5 text encoder...")
        self.text_encoder, self.tokenizer = self._load_text_encoder()

        dit_dim = self.wan.config.dim 

        print("  [4/5] Creating ControlAdapter...")
        self.control_adapter = ControlAdapter(
            control_dim=256,
            hidden_dim=512,
            dit_dim=dit_dim,
            num_controls=1,
            use_gradient_checkpointing=False,
        ).to(device)

        print("  [5/5] Creating zero-conv projections (one per injection layer)...")
        print(f"dit_dim = {self.wan.config.dim}")
        self.zero_convs = nn.ModuleList([
            ZeroLinear(dit_dim) for _ in control_injection_layers
        ]).to(device)

      
        self._block_to_hook_idx: dict[int, int] = {}
        self._setup_control_hooks()

      
        self._control_signal: torch.Tensor | None = None

        
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        zc_params = sum(p.numel() for p in self.zero_convs.parameters())

        print(f"{'='*70}")
        print(f"  Total params:              {total:,} ({total / 1e9:.2f}B)")
        print(f"  Frozen (WAN + VAE + T5):   {frozen:,} ({frozen / 1e9:.2f}B)")
        print(f"  Trainable — Adapter:       {sum(p.numel() for p in self.control_adapter.parameters()):,}")
        print(f"  Trainable — Zero convs:    {zc_params:,}  ({len(self.zero_convs)} layers × {zc_params // len(self.zero_convs):,})")
        print(f"  Trainable — Total:         {trainable:,} ({trainable / 1e6:.1f}M)")
        print(f"  Injection layers:          {control_injection_layers}")
        print(f"{'='*70}\n")

    def _setup_control_hooks(self):
        self._current_wan_grid = None
        self._grid_hook = self.wan.patch_embedding.register_forward_hook(
            self._capture_wan_grid_hook
        )

        self.hooks = []
        for hook_idx, layer_idx in enumerate(self.control_injection_layers):
            if layer_idx >= len(self.wan.blocks):
                continue
            block = self.wan.blocks[layer_idx]
            self._block_to_hook_idx[id(block)] = hook_idx
            hook = block.register_forward_pre_hook(self._control_injection_hook)
            self.hooks.append(hook)


    def _capture_wan_grid_hook(self, module, inputs, output):
        """Capture WAN's real post-patch temporal and spatial grid."""
        self._current_wan_grid = tuple(
            int(value) for value in output.shape[2:]
        )

    def _control_injection_hook(self, module, input):
        if self._control_signal is None:
            return input

        x = input[0]
        B, L, C = x.shape
        hook_idx = self._block_to_hook_idx[id(module)]
        zero_conv = self.zero_convs[hook_idx]
        ctrl = self._control_signal  

        if ctrl.shape[1] != L:
            B_c, S_c, C_c = ctrl.shape

            if S_c % (16 * 16) != 0:
                raise RuntimeError(
                    f"Invalid control token count: {S_c}"
                )

            if self._current_wan_grid is None:
                raise RuntimeError("WAN patch grid was not captured")

            T_c = S_c // (16 * 16)
            T_real, H_real, W_real = self._current_wan_grid
            real_length = T_real * H_real * W_real

            if real_length > L:
                raise RuntimeError(
                    f"WAN grid has {real_length} tokens but L={L}"
                )

            ctrl = ctrl.reshape(
                B_c, T_c, 16, 16, C_c
            ).permute(0, 4, 1, 2, 3)

            ctrl = F.interpolate(
                ctrl.float(),
                size=(T_real, H_real, W_real),
                mode="trilinear",
                align_corners=False,
            )

            ctrl = ctrl.permute(
                0, 2, 3, 4, 1
            ).reshape(B_c, real_length, C_c)

            if real_length < L:
                padding = ctrl.new_zeros(
                    B_c, L - real_length, C_c
                )
                ctrl = torch.cat([ctrl, padding], dim=1)

            assert ctrl.shape[1] == L

        ctrl = zero_conv(ctrl)

     
        x_norm = x.norm(dim=-1, keepdim=True).mean()
        ctrl_norm = ctrl.norm(dim=-1, keepdim=True).mean()
        if ctrl_norm > 0:
            ctrl = ctrl * (x_norm / ctrl_norm).clamp(max=1.0) * 0.1

        # Default 1.0 preserves the checkpoint's original behavior.
        control_strength = float(
            getattr(self, "_control_strength", 1.0)
        )
        ctrl = ctrl * control_strength

        x = x + ctrl
        return (x,) + input[1:]

    def _load_vae(self):
        from wan.modules.vae2_2 import Wan2_2_VAE

        vae_path = self.checkpoint_dir / 'Wan2.2_VAE.pth'
        if not vae_path.exists():
            raise FileNotFoundError(f"VAE not found at {vae_path}")

        print(f"     Loading VAE from {vae_path}...")
        device_vae = torch.device('cuda:1')
        vae = Wan2_2_VAE(
            vae_pth=str(vae_path),
            z_dim=48,
            c_dim=160,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
            dtype=torch.float16,
            device=device_vae,
        )
        for param in vae.model.parameters():
            param.requires_grad = False
        print("   VAE loaded successfully")
        return vae

    def _load_wan(self):
        from safetensors.torch import load_file
        import json

        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path) as f:
            model_config = json.load(f)

        print(f"     Model config: {model_config.get('num_layers', 32)} layers")

        wan = WanModel(
            model_type=model_config.get('model_type', 'ti2v'),
            patch_size=tuple(model_config.get('patch_size', [1, 2, 2])),
            text_len=model_config.get('text_len', 512),
            in_dim=model_config.get('in_dim', 16),
            dim=model_config.get('dim', 2048),
            ffn_dim=model_config.get('ffn_dim', 8192),
            freq_dim=model_config.get('freq_dim', 256),
            text_dim=model_config.get('text_dim', 4096),
            out_dim=model_config.get('out_dim', 16),
            num_heads=model_config.get('num_heads', 16),
            num_layers=model_config.get('num_layers', 32),
            window_size=tuple(model_config.get('window_size', [-1, -1])),
            qk_norm=model_config.get('qk_norm', True),
            cross_attn_norm=model_config.get('cross_attn_norm', True),
            eps=model_config.get('eps', 1e-6),
        )

        print("     Loading WAN weights from safetensors...")
        index_path = self.checkpoint_dir / 'diffusion_pytorch_model.safetensors.index.json'
        with open(index_path) as f:
            index = json.load(f)

        state_dict = {}
        for shard_file in sorted(set(index['weight_map'].values())):
            shard_path = self.checkpoint_dir / shard_file
            print(f"     Loading {shard_file}...")
            state_dict.update(load_file(str(shard_path)))

        wan.load_state_dict(state_dict)
        wan = wan.to(torch.bfloat16)
        wan = wan.eval()

        for param in wan.parameters():
            param.requires_grad = False

        

        print(f"   WAN loaded ({model_config.get('num_layers', 32)} layers)")
        return wan

    def _load_text_encoder(self):
        from wan.modules.t5 import T5EncoderModel

        t5_path = self.checkpoint_dir / 'models_t5_umt5-xxl-enc-bf16.pth'
        if not t5_path.exists():
            raise FileNotFoundError(f"T5 weights not found at {t5_path}")

        print(f"     Loading T5 from {t5_path}...")
        device_cpu = 'cpu'
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device_cpu,
            checkpoint_path=str(t5_path),
            tokenizer_path='google/umt5-xxl',
        )
        self.t5_device = device_cpu
        print(f"  T5 encoder loaded on {device_cpu} (to save GPU memory)")
        return text_encoder, text_encoder.tokenizer

    

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae_device = torch.device('cuda:1')
        output_device = torch.device('cuda:0')
        video = video.to(vae_device)
        self.vae.device = vae_device
        with torch.no_grad():
            latents = self.vae.encode([video[i] for i in range(video.shape[0])])
            latent = torch.stack(latents)
        return latent.to(output_device)

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        vae_device = torch.device('cuda:1')
        output_device = torch.device('cuda:0')
        latent = latent.to(vae_device)
        self.vae.device = vae_device
        with torch.no_grad():
            videos = self.vae.decode([latent[i] for i in range(latent.shape[0])])
            video = torch.stack(videos)
        return video.to(output_device)

    def encode_text(self, prompts: list) -> list:
        t5_compute_device = (
            torch.device('cuda:1') if torch.cuda.device_count() > 1
            else torch.device('cuda:0')
        )
        with torch.no_grad():
            self.text_encoder.model.to(t5_compute_device)
            text_list = self.text_encoder(prompts, device=t5_compute_device)
            self.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        padded = []
        for t in text_list:
            if t.shape[0] < 512:
                padding = torch.zeros(
                    512 - t.shape[0], t.shape[1],
                    device=t.device, dtype=t.dtype,
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t.to(self.device, dtype=torch.float32))
        return padded

    
    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: list,
        control_features: dict = None,
    ) -> torch.Tensor:

        offload_model = True


        if control_features is not None:
            t0 = time.time()
            # controls_device = {k: v.to(self.device) for k, v in control_features.items()}
            controls_device = control_features
            print(f"  Control move:    {time.time() - t0:.1f}s")

            t0 = time.time()
           
            self._control_signal = self.control_adapter(controls_device)
            del controls_device
            torch.cuda.empty_cache()

            print(f"  Control adapter: {time.time() - t0:.1f}s")
        else:
            self._control_signal = None


        torch.cuda.empty_cache()
        if offload_model:
            t0 = time.time()
            self.wan.to(self.device)
            torch.cuda.empty_cache()
            print(f"  WAN load:        {time.time() - t0:.1f}s")

        latent = latent.to(self.device)
        timesteps = timesteps.to(self.device)


        t0 = time.time()
        print('encoding text')
        if isinstance(prompts, torch.Tensor):
          
            text_embeddings = [prompts[i].to(self.device, dtype=torch.float32) 
                            for i in range(prompts.shape[0])]
        elif isinstance(prompts, (list, tuple)) and isinstance(prompts[0], torch.Tensor):
         
            text_embeddings = [p.to(self.dit_device, dtype=torch.float32) for p in prompts]
        else:
            
            text_embeddings = self.encode_text(prompts)
        
        context = text_embeddings
        x = [latent[i] for i in range(latent.shape[0])]

        

        B, C, T, H, W = latent.shape
        patch_t= T // self.wan.patch_size[0]
        patch_h= H // self.wan.patch_size[1]
        patch_w = W // self.wan.patch_size[2]
     
        seq_len_actual = patch_t * patch_h * patch_w
        seq_len = ((seq_len_actual + 63) // 64) * 64
        print(seq_len, 'seq len')
        t0 = time.time()
       
        noise_pred = self.wan(
                x=x,
                t=timesteps,
                context=context,
                seq_len=seq_len,
                y=None,
            )
        print(f"  WAN forward:     {time.time() - t0:.1f}s")

        noise_pred = torch.stack(list(noise_pred))
        self._control_signal = None   


        
        torch.cuda.empty_cache()

        return noise_pred



    def get_trainable_parameters(self) -> list:
       
        return [p for p in self.parameters() if p.requires_grad]

    def get_trainable_parameter_groups(self) -> list[dict]:
      
        gate_params = [self.control_adapter.modality_gates]
        adapter_params = [p for n, p in self.control_adapter.named_parameters() 
                        if 'modality_gates' not in n]
        return [
            {'params': adapter_params,          'name': 'control_adapter'},
            {'params': list(self.zero_convs.parameters()), 'name': 'zero_convs'},
            {'params': gate_params,             'name': 'modality_gates'},
        ]



def test_controllable_wan():
    print("\n" + "=" * 70)
    print("Testing Controllable WAN  (zero-conv injection)")
    print("=" * 70)

    model = ControllableWAN(
        checkpoint_dir='Wan2.2/Wan2.2-TI2V-5B',
        device='cuda',
    )

    B = 1
    latent = torch.randn(B, 48, 8, 32, 32).cuda()
    timesteps = torch.randint(0, 1000, (B,)).cuda()
    prompts = ["A cat playing with a ball"]

    controls = {
        'depth_encoded':  torch.randn(B, 256, 8, 128, 128).cuda(),
        'sketch_encoded': torch.randn(B, 256, 8, 128, 128).cuda(),
        'motion_encoded': torch.randn(B, 256, 8, 128, 128).cuda(),
        'style_encoded':  torch.randn(B, 256, 8,  32,  32).cuda(),
        'pose_encoded':   torch.randn(B, 256, 8, 128, 128).cuda(),
        'mask_encoded':   torch.randn(B, 256, 8, 128, 128).cuda(),
    }

  
    print("\nVerifying zero-conv guarantee...")
    ctrl_signal = model.control_adapter(
        {k: v.to(model.device) for k, v in controls.items()}
    )
    for i, zc in enumerate(model.zero_convs):
        out = zc(ctrl_signal)
        assert out.abs().max().item() == 0.0, f"zero_conv[{i}] is not zero at init!"
    print("  ✓  All zero convs output exactly 0.0 at initialisation")

    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(latent, timesteps, prompts, controls)

    total_trainable = sum(p.numel() for p in model.get_trainable_parameters())
    zc_trainable    = sum(p.numel() for p in model.zero_convs.parameters())

    print(f"\nForward pass successful!")
    print(f"  Input latent:          {latent.shape}")
    print(f"  Output:                {output.shape}")
    print(f"  Trainable total:       {total_trainable:,}")
    print(f"    — ControlAdapter:    {total_trainable - zc_trainable:,}")
    print(f"    — Zero convs:        {zc_trainable:,}  ({len(model.zero_convs)} × {zc_trainable // len(model.zero_convs):,})")


if __name__ == '__main__':
    test_controllable_wan()