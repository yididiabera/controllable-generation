import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import FlowMatchEulerDiscreteScheduler
from pathlib import Path
import sys
from tqdm import tqdm
import json
import gc

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Wan2.2'))

from data.dataset import ControllableVideoDataset
from depth_models.wan_controllable import ControllableWAN



def flow_matching_loss(noise_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    return F.mse_loss(noise_pred, target, reduction='mean')


def timestep_weighted_flow_loss(
    noise_pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    
    t_norm = timesteps.float() / 1000.0                    
    weights = 1.0 - 0.5 * t_norm                           
    weights = weights.view(-1, 1, 1, 1, 1)                 

    per_element_loss = (noise_pred - target) ** 2          
    weighted_loss = (per_element_loss * weights).mean()
    return weighted_loss


def temporal_smoothness_loss(
    noise_pred: torch.Tensor,
) -> torch.Tensor:
    """Mean squared difference between adjacent latent frames."""
    if noise_pred.dim() != 5:
        raise ValueError(
            f"Expected [B,C,T,H,W] prediction, got {noise_pred.shape}"
        )
    if noise_pred.shape[2] <= 1:
        return noise_pred.new_zeros(())
    return (
        noise_pred[:, :, 1:] - noise_pred[:, :, :-1]
    ).square().mean()


def control_adherence_loss(
    model: 'ControllableWAN',
    noisy_latent: torch.Tensor,
    timesteps: torch.Tensor,
    prompts: list,
    control_features: dict,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    with torch.no_grad():
        pred_no_ctrl = model(
            latent=noisy_latent,
            timesteps=timesteps,
            prompts=prompts,
            control_features=None,
        )

    loss_with_ctrl  = F.mse_loss(model._last_noise_pred, target)
    loss_no_ctrl    = F.mse_loss(pred_no_ctrl, target)

  
    adherence = loss_no_ctrl - loss_with_ctrl
    adherence_loss = -adherence.clamp(min=-1.0) 

    return adherence_loss, adherence.item()


class MultiVideoTrainer:

    def __init__(
        self,
        model: ControllableWAN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
        resume_from: str = None,
    ):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        # if torch.cuda.device_count() > 1:
        #     print(f"\n  Using {torch.cuda.device_count()} GPUs!")
        #     self.model      = nn.DataParallel(model)
        #     self.base_model = model
        # else:
        #     print("\n  Single GPU detected")
        #     self.model      = model
        #     self.base_model = model
        print(f"\n  GPUs: {torch.cuda.device_count()} (cuda:0=WAN+adapter, cuda:1=VAE)")
        self.model = model
        self.base_model = model

        param_groups = self.base_model.get_trainable_parameter_groups()
       
        for g in param_groups:
            if g['name'] == 'zero_convs':
                g['lr'] = config['lr'] * 2.0
            elif g['name'] == 'modality_gates': 
                g['lr'] = config['lr'] * 20.0
            else:
                g['lr'] = config['lr']

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_steps'],
            eta_min=config['lr'] * 0.1,
        )

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config['mixed_precision'])

        self.global_step   = 0
        self.start_epoch   = 0
        self.best_val_loss = float('inf')

        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / 'training_log.jsonl'

        if resume_from:
            self.load_checkpoint(resume_from)

        trainable_params = self.base_model.get_trainable_parameters()
        zc_params = sum(p.numel() for p in self.base_model.zero_convs.parameters())

        print(f"\n{'='*70}")
        print("Multi-Video Trainer Initialized")
        print(f"{'='*70}")
        print(f"  Starting step/epoch:    {self.global_step} / {self.start_epoch}")
        print(f"  Best val loss:          {self.best_val_loss:.4f}")
        print(f"  GPUs:                   {torch.cuda.device_count()}")
        print(f"  Optimizer:              AdamW")
        print(f"    Adapter LR:           {config['lr']}")
        print(f"    Zero conv LR:         {config['lr'] * 2.0}  (2× for zero-init break)")
        print(f"  Mixed Precision:        {config['mixed_precision']}")
        print(f"  Grad accumulation:      {config['grad_accum_steps']} steps")
        print(f"  Effective batch size:   {config['batch_size'] * config['grad_accum_steps']}")
        print(f"  Loss weights:")
        print(f"    Flow matching:        {config.get('loss_flow_weight', 1.0)}")
        print(f"    Timestep-weighted:    {config.get('loss_weighted_weight', 0.1)}")
        print(f"    Control adherence:    {config.get('loss_adherence_weight', 0.0)}  (0=off)")
        print(f"  Trainable params:       {sum(p.numel() for p in trainable_params):,}")
        print(f"    — Adapter:            {sum(p.numel() for p in trainable_params) - zc_params:,}")
        print(f"    — Zero convs:         {zc_params:,}")
        print(f"{'='*70}\n")

    

    def load_checkpoint(self, checkpoint_path: str):
        print(f"\n  Resuming from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.base_model.control_adapter.load_state_dict(checkpoint['model'])
        print("    Loaded adapter weights")

        if 'zero_convs' in checkpoint:
            self.base_model.zero_convs.load_state_dict(checkpoint['zero_convs'])
            print("    Loaded zero conv weights")
        else:
            print("    WARNING: no zero_conv weights in checkpoint (old format)")

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            print("    Loaded optimizer state")
        except ValueError as e:
            print(f"    WARNING: Could not load optimizer state ({e})")
            print("    Starting optimizer fresh (weights still loaded)")

        self.global_step   = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.start_epoch   = checkpoint.get('epoch', 0)
        print(f"    Step {self.global_step}, epoch {self.start_epoch}, best loss {self.best_val_loss:.4f}\n")

    def save_checkpoint(self, name: str, epoch: int = 0):
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{name}.pt'
        torch.save({
           
            'model':      self.base_model.control_adapter.state_dict(),
            'zero_convs': self.base_model.zero_convs.state_dict(),   
           
            'optimizer':    self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler':       self.scaler.state_dict(),
            
            'global_step':   self.global_step,
            'epoch':         epoch,
            'best_val_loss': self.best_val_loss,
            'config':        self.config,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def export_for_inference(self, name: str = 'final'):
        from safetensors.torch import save_file

      
        adapter_path = self.checkpoint_dir / f'control_adapter_{name}.safetensors'
        save_file(
            self.base_model.control_adapter.state_dict(),
            str(adapter_path),
            metadata={
                'model_type': 'ControlAdapter',
                'training_step': str(self.global_step),
                'best_val_loss': str(self.best_val_loss),
            }
        )

       
        zc_path = self.checkpoint_dir / f'zero_convs_{name}.safetensors'
        save_file(
            self.base_model.zero_convs.state_dict(),
            str(zc_path),
            metadata={
                'model_type': 'ZeroConvs',
                'num_layers': str(len(self.base_model.zero_convs)),
                'training_step': str(self.global_step),
            }
        )
        print(f"  Exported adapter: {adapter_path} ({adapter_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  Exported zero convs: {zc_path} ({zc_path.stat().st_size / 1e6:.1f} MB)")


    def get_zero_conv_stats(self) -> dict:
    
        stats = {}
        for i, zc in enumerate(self.base_model.zero_convs):
            w_norm = zc.proj.weight.abs().mean().item()
            b_norm = zc.proj.bias.abs().mean().item()
            stats[f'zero_conv_{i}_weight_norm'] = w_norm
            stats[f'zero_conv_{i}_bias_norm']   = b_norm

        
        all_w = [v for k, v in stats.items() if 'weight' in k]
        stats['zero_conv_mean_weight_norm'] = sum(all_w) / len(all_w)
        stats['zero_conv_max_weight_norm']  = max(all_w)
        stats['zero_conv_min_weight_norm']  = min(all_w)
        return stats

    def get_modality_gate_stats(self) -> dict:
        
        gates = self.base_model.control_adapter.get_modality_weights()
        return {f'gate_{k}': v for k, v in gates.items()}

   

    def train_step(self, batch) -> tuple[torch.Tensor, dict]:
        video    = batch['video'].to(self.device)
        controls = {k: v.to(self.device) for k, v in batch['controls'].items()}
        active_controls = ['depth_encoded']
        controls = {k: v for k, v in controls.items() if k in active_controls}
        # text_embeddings = batch['caption'].to(self.device)
        caption = batch['caption']
        if isinstance(caption, torch.Tensor):
            text_embeddings = caption.to(self.device)
        else:
            # raw strings from DataLoader come as a list
            text_embeddings = list(caption)

        with torch.no_grad():
            latent = self.base_model.encode_video(video)
        del video

        print(f"  latent shape: {latent.shape}")
        torch.cuda.empty_cache()

        B = latent.shape[0]
        timesteps = torch.randint(50, 950, (B,), device=self.device).long()
        noise     = torch.randn_like(latent, dtype=torch.float32)
        t         = (timesteps.float() / 1000.0).view(-1, 1, 1, 1, 1)
        noisy     = ((1 - t) * latent + t * noise)
        target    = (noise - latent)
        del t, noise
        del latent
        torch.cuda.empty_cache()
       

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self.config['mixed_precision']):
            noise_pred = self.base_model(
                latent=noisy,
                timesteps=timesteps,
                prompts=text_embeddings,
                control_features=controls,
            )

        w_flow = self.config.get('loss_flow_weight', 1.0)
        w_weighted = self.config.get(
            'loss_weighted_weight', 0.1
        )
        w_temporal = self.config.get(
            'loss_temporal_weight', 0.05
        )

        loss_flow = flow_matching_loss(noise_pred, target)
        loss_weighted = timestep_weighted_flow_loss(
            noise_pred, target, timesteps
        )
        loss_temporal = temporal_smoothness_loss(noise_pred)
        total_loss = (
            w_flow * loss_flow
            + w_weighted * loss_weighted
            + w_temporal * loss_temporal
        )

      
        gates = torch.sigmoid(self.base_model.control_adapter.modality_gates)
        gate_entropy = -(gates * torch.log(gates + 1e-8) +
                        (1 - gates) * torch.log(1 - gates + 1e-8)).mean()

        # total_loss = total_loss + 0.01 * gate_entropy

        del noisy, noise_pred, controls
        torch.cuda.empty_cache()

        metrics = {
            'loss':            total_loss.item(),
            'loss_flow':       loss_flow.item(),
            'loss_weighted':   loss_weighted.item(),
            'loss_temporal':   loss_temporal.item(),
            'loss_temporal_weighted': (
                w_temporal * loss_temporal
            ).item(),
            'adherence_delta': 0.0,   
            'lr_adapter':      self.optimizer.param_groups[0]['lr'],
            'lr_zero_conv':    self.optimizer.param_groups[1]['lr'],
            'timestep_mean':   timesteps.float().mean().item(),
        }
        return total_loss, metrics
   
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_losses = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            try:
                loss, metrics = self.train_step(batch)

                loss_scaled = loss / self.config['grad_accum_steps']
                self.scaler.scale(loss_scaled).backward()
                epoch_losses.append(metrics['loss'])

                progress_bar.set_postfix({
                    'loss':   f"{metrics['loss']:.4f}",
                    'avg':    f"{sum(epoch_losses)/len(epoch_losses):.4f}",
                    'adhere': f"{metrics['adherence_delta']:+.4f}",
                    'gpu0':   f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB",
                    'step':   self.global_step,
                })

                if (step + 1) % self.config['grad_accum_steps'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.base_model.get_trainable_parameters(),
                        self.config['max_grad_norm'],
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    self.global_step += 1
                    torch.cuda.empty_cache()

                    if self.global_step % self.config['log_every'] == 0:
                        log_data = {
                            'step':    self.global_step,
                            'epoch':   epoch,
                            **metrics,
                            'gpu0_memory_gb': torch.cuda.memory_allocated(0)/1e9,
                        }
                        if torch.cuda.device_count() > 1:
                            log_data['gpu1_memory_gb'] = torch.cuda.memory_allocated(1)/1e9

                        
                        log_data.update(self.get_zero_conv_stats())

                       
                        log_data.update(self.get_modality_gate_stats())

                        self.log_metrics(log_data)

                       
                        zc_stats = self.get_zero_conv_stats()
                        print(
                            f"\n  [step {self.global_step}] "
                            f"loss={metrics['loss']:.4f}  "
                            f"adherence={metrics['adherence_delta']:+.4f}  "
                            f"zc_mean_norm={zc_stats['zero_conv_mean_weight_norm']:.6f}  "
                            f"zc_max_norm={zc_stats['zero_conv_max_weight_norm']:.6f}"
                        )

                    if self.global_step % self.config['save_every'] == 0:
                        self.save_checkpoint(f'step_{self.global_step}', epoch=epoch)

                    if self.global_step % self.config['val_every'] == 0:
                        val_loss = self.validate()
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint('best', epoch=epoch)
                            print(f"  New best val loss: {val_loss:.4f}")
                        self.model.train()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  OOM at step {step}, skipping batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.optimizer.zero_grad()
                    continue
                raise e

        return sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')

   
    @torch.no_grad()
    def val_step(self, batch) -> dict:
        video = batch['video'].to(self.device)
        controls = {k: v.to(self.device) for k, v in batch['controls'].items()}
        text_embeddings = batch['caption'].to(self.device)

        
        latent = self.base_model.encode_video(video)
        del video
        torch.cuda.empty_cache()

        B = latent.shape[0]

       
        timesteps = torch.randint(0, 1000, (B,), device=self.device).long()
        noise = torch.randn_like(latent, dtype=torch.float32)

        t = (timesteps.float() / 1000.0).view(-1, 1, 1, 1, 1)
        noisy = (1 - t) * latent + t * noise
        target = noise - latent

        del latent, noise

   
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, 
                            enabled=self.config['mixed_precision']):
            noise_pred = self.base_model(
                latent=noisy,
                timesteps=timesteps,
                prompts=text_embeddings,
                control_features=controls,
            )

        loss_flow = flow_matching_loss(noise_pred, target)
        loss_weighted = timestep_weighted_flow_loss(
            noise_pred, target, timesteps
        )
        loss_temporal = temporal_smoothness_loss(noise_pred)

        w_flow = self.config.get('loss_flow_weight', 1.0)
        w_weighted = self.config.get(
            'loss_weighted_weight', 0.1
        )
        w_temporal = self.config.get(
            'loss_temporal_weight', 0.05
        )
        total_loss = (
            w_flow * loss_flow
            + w_weighted * loss_weighted
            + w_temporal * loss_temporal
        )

        del noisy, noise_pred, controls

        return {
            'loss': total_loss.item(),
            'loss_flow': loss_flow.item(),
            'loss_weighted': loss_weighted.item(),
            'loss_temporal': loss_temporal.item(),
            'loss_temporal_weighted': (
                w_temporal * loss_temporal
            ).item(),
        }

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        val_losses = []
        print("\n  Running validation...")

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            try:
                metrics = self.val_step(batch)  
                val_losses.append(metrics['loss'])
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise e

        avg_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        print(f"  Validation Loss: {avg_loss:.4f}")
        self.log_metrics({'step': self.global_step, 'val_loss': avg_loss})
        return avg_loss

    def log_metrics(self, metrics: dict):
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')



def main():
    print("\n" + "=" * 70)
    print("Training Controllable WAN - Multiple Videos")
    print("=" * 70)

    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    config = {
        'batch_size':       1,
        'num_workers':      0,

        'num_epochs':       40,
        'num_steps':        50000,
        'grad_accum_steps': 8,
        'mixed_precision':  True,

        'num_frames':       8,
        'resolution':       (128, 128),
        'control_resolution': (16, 16),

        'lr':               1e-4,
        'weight_decay':     0.01,
        'max_grad_norm':    1.0,

        'loss_flow_weight':      1.0,
        'loss_weighted_weight':  0.1,
        'loss_temporal_weight':  0.05,
        'loss_adherence_weight': 0.00,  
        'log_every':   10,
        'save_every':  100,
        'val_every':   500,

        'checkpoint_dir':  '/mnt/d1/jedidiah/checkpoints/depth_grid_fixed_docx',
        'data_dir':        '/mnt/d1/jedidiah/data_depth_repro_docx',
        'checkpoint_path': '/mnt/d1/jedidiah/models/Wan2.2-TI2V-5B',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    gc.collect()

    resume_checkpoint = None
    checkpoint_dir = Path(config['checkpoint_dir'])
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"\n  Found checkpoint: {latest.name}")
            if input("  Resume training? (y/n): ").lower() == 'y':
                resume_checkpoint = str(latest)
    print("\n[1/4] Loading Controllable WAN...")
    model = ControllableWAN(checkpoint_dir=config['checkpoint_path'], device=device)
    print("\n[2/4] Loading datasets...")
    full_dataset = ControllableVideoDataset(
        encoded_controls_dir=f"{config['data_dir']}/encoded_controls",
        videos_dir=f"{config['data_dir']}/videos",
        annotations_path=f"{config['data_dir']}/shots_metadata.json",
        num_frames=config['num_frames'],
        resolution=config['resolution'],
        split='train',
        text_encoder=model,
        load_videos=True,
    )
    val_dataset = ControllableVideoDataset(
        encoded_controls_dir=f"{config['data_dir']}/encoded_controls",
        videos_dir=f"{config['data_dir']}/videos",
        annotations_path=f"{config['data_dir']}/shots_metadata.json",
        num_frames=config['num_frames'],
        resolution=config['resolution'],
        split='val',
        text_encoder=model,
        load_videos=True,
    )
    print(f"  Train: {len(full_dataset)} samples  |  Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        full_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=False,
    )

   

    print("\n[3/4] Creating trainer...")
    trainer = MultiVideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        resume_from=resume_checkpoint,
    )


    with torch.no_grad():
        trainer.base_model.control_adapter.modality_gates.copy_(
        torch.randn(1) * 0.1
    )
    print("Gates reset:", torch.sigmoid(trainer.base_model.control_adapter.modality_gates))

    print("\n[4/4] Starting training...")
    print("=" * 70)
    print("  Watch 'zero_conv_mean_weight_norm' in logs.")
    print("  It should start at 0.0 and slowly rise over thousands of steps.")
    print("  If it stays at 0.0 after 500 steps → gradients not flowing → bug.")
    print("  If it jumps to >0.1 in first 100 steps → LR too high.")
    print("=" * 70 + "\n")

    try:
        for epoch in range(trainer.start_epoch, config['num_epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{config['num_epochs']}  (step {trainer.global_step})")
            print(f"{'='*70}")

            avg_loss = trainer.train_epoch(epoch)

            zc_stats = trainer.get_zero_conv_stats()
            gate_stats = trainer.get_modality_gate_stats()

            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Avg loss:            {avg_loss:.4f}")
            print(f"    Global step:         {trainer.global_step}")
            print(f"    Zero conv mean norm: {zc_stats['zero_conv_mean_weight_norm']:.6f}")
            print(f"    Zero conv max norm:  {zc_stats['zero_conv_max_weight_norm']:.6f}")
            print(f"    Modality gates:      { {k.replace('gate_',''):f'{v:.3f}' for k,v in gate_stats.items()} }")

            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint(f'epoch_{epoch+1}', epoch=epoch+1)
                trainer.export_for_inference(f'epoch_{epoch+1}')

            if trainer.global_step >= config['num_steps']:
                print(f"\n  Reached max steps ({config['num_steps']})")
                break

        final_val_loss = trainer.validate()
        trainer.save_checkpoint('final', epoch=config['num_epochs'])
        trainer.export_for_inference('final')

        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"  Best val loss:   {trainer.best_val_loss:.4f}")
        print(f"  Final val loss:  {final_val_loss:.4f}")
        print(f"  Total steps:     {trainer.global_step}")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n  Training interrupted")
        trainer.save_checkpoint('interrupted', epoch=epoch)
        trainer.export_for_inference('interrupted')

    except Exception as e:
        print(f"\n  Training failed: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint('error', epoch=epoch)


if __name__ == '__main__':
    main()
