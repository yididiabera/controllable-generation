
import sys
import gc
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Wan2.2'))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from src.depth_models.wan_controllable import ControllableWAN



def load_midas(device, models_dir, midas_weights=None, allow_download=False):
    """Load MiDaS the same way training control extraction does."""
    models_dir = Path(models_dir)
    if not models_dir.is_absolute():
        models_dir = project_root / models_dir

    weights_path = Path(midas_weights) if midas_weights else models_dir / "midas_v3_dpt_large.pth"
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path

    model_type = "DPT_Large"
    print("  Loading MiDaS DPT_Large...")
    if weights_path.exists():
        print(f"    Using local weights: {weights_path}")
        try:
            midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False)
            checkpoint = torch.load(weights_path, map_location=device)
            midas.load_state_dict(checkpoint)
        except RuntimeError as e:
            print("    WARNING: local MiDaS weights are incompatible with DPT_Large")
            print(f"    {str(e).splitlines()[0]}")
            print("    Falling back to torch.hub pretrained MiDaS")
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
    elif allow_download:
        print(f"    WARNING: local weights not found at {weights_path}")
        print("    Falling back to torch.hub pretrained MiDaS")
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
    else:
        raise FileNotFoundError(
            f"MiDaS weights not found at {weights_path}. "
            "Pass --midas_weights, --models_dir, or --allow_midas_download."
        )

    midas.to(device).eval()
    tf = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return midas, tf


def extract_depth_frame(frame_bgr, midas, tf, device, midas_hw=(360, 640), control_hw=(128, 128)):
    """
    Match training raw-MiDaS controls:
    frame -> MiDaS -> per-frame uint8 depth at 360x640 -> resize to control_hw -> [0, 1].
    """
    inp = tf(frame_bgr).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = F.interpolate(pred.unsqueeze(1), size=midas_hw,
                             mode="bicubic", align_corners=False).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d = (d * 255).astype(np.uint8)
    d = cv2.resize(d, (control_hw[1], control_hw[0]), interpolation=cv2.INTER_LINEAR)
    return d.astype(np.float32) / 255.0


def extract_depth_sequence(
    video_path,
    midas,
    tf,
    device,
    num_frames=8,
    midas_hw=(360, 640),
    control_hw=(128, 128),
):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frames from {video_path}")
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
    depths = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            depths.append(depths[-1] if depths else np.zeros(control_hw, np.float32))
            continue
        depths.append(extract_depth_frame(frame, midas, tf, device, midas_hw, control_hw))
    cap.release()
    return np.stack(depths) 

def depth_to_control_tensor(depth_seq, device):
    """(T, H, W) → (1, 256, T, H, W) float32"""
    t = torch.from_numpy(depth_seq).float()
    t = t.unsqueeze(0).unsqueeze(0)           
    t = t.expand(-1, 256, -1, -1, -1).clone()
    return t.to(device)


def activate_adapter(controllable_wan, control_features):
    """Compute control signal and set it so hooks fire on next WAN forward."""
    with torch.no_grad():
        ctrl = controllable_wan.control_adapter(control_features)
    controllable_wan._control_signal = ctrl
    print(f"  Adapter activated  |  ctrl norm={ctrl.norm():.4f}  "
          f"gate={torch.sigmoid(controllable_wan.control_adapter.modality_gates).item():.4f}")


def deactivate_adapter(controllable_wan):
    """Clear control signal — WAN runs without adapter."""
    controllable_wan._control_signal = None


def describe_video_tensor(name, video_tensor):
    v = video_tensor.detach().float()
    print(
        f"  {name} tensor: shape={tuple(video_tensor.shape)} "
        f"dtype={video_tensor.dtype} device={video_tensor.device} "
        f"min={v.nan_to_num().min().item():.4f} "
        f"max={v.nan_to_num().max().item():.4f} "
        f"mean={v.nan_to_num().mean().item():.4f} "
        f"nan={torch.isnan(v).any().item()} inf={torch.isinf(v).any().item()}"
    )


def video_to_bcthw(video_tensor):
    """Normalize WAN output variants to (B, C, T, H, W)."""
    if video_tensor.dim() == 5:
        if video_tensor.shape[1] in (1, 3):
            return video_tensor
        if video_tensor.shape[2] in (1, 3):
            return video_tensor.permute(0, 2, 1, 3, 4).contiguous()
    elif video_tensor.dim() == 4:
        if video_tensor.shape[0] in (1, 3):
            return video_tensor.unsqueeze(0)
        if video_tensor.shape[1] in (1, 3):
            return video_tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    raise ValueError(f"Unexpected video tensor shape: {tuple(video_tensor.shape)}")


def tensor_to_frames(video_tensor):
    """
    WAN generate() returns video in [-1, 1].
    Returns (T, H, W, 3) uint8 RGB.
    """
    v = video_to_bcthw(video_tensor)[0].float()
    v = (v.clamp(-1, 1) + 1) / 2 * 255
    v = v.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    return v


def save_rgb_video(frames, path, fps=16):
    """Save known-good RGB uint8 frames with a broadly compatible H.264 encode."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames (T, H, W, 3), got {frames.shape}")

    # yuv420p H.264 needs even dimensions. Edge padding avoids resizing artifacts.
    pad_h = frames.shape[1] % 2
    pad_w = frames.shape[2] % 2
    if pad_h or pad_w:
        frames = np.pad(
            frames,
            ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode='edge',
        )
    frames = np.ascontiguousarray(frames)

    try:
        import imageio.v2 as imageio
        imageio.mimsave(
            str(path),
            list(frames),
            fps=fps,
            codec='libx264',
            quality=8,
            macro_block_size=1,
            ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'],
        )
        print(f"    Saved: {path}  ({len(frames)} RGB frames @ {fps} fps, libx264/yuv420p)")
    except Exception as e:
        fallback_path = Path(path).with_suffix('.avi')
        print(f"    WARNING: H.264 encode failed ({e}); writing MJPG AVI: {fallback_path}")
        T, H, W, _ = frames.shape
        writer = cv2.VideoWriter(
            str(fallback_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H)
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"    Saved fallback: {fallback_path}  ({T} frames @ {fps} fps)")


def save_debug_frames(frames, output_dir, prefix):
    for idx in sorted({0, len(frames) // 2, len(frames) - 1}):
        path = output_dir / f"{prefix}_frame_{idx:03d}.png"
        Image.fromarray(frames[idx]).save(path)
        print(f"    Saved debug frame: {path}")


def save_comparison_video(frames, path, fps=16):
    save_rgb_video(frames, path, fps=fps)


def depth_to_rgb(depth_seq, target_hw):
    """(T, H, W) float32 → (T, H, W, 3) uint8 plasma, resized to target_hw."""
    H, W = target_hw
    frames = []
    for d in depth_seq:
        d_u8 = (d * 255).astype(np.uint8)
        col  = cv2.applyColorMap(d_u8, cv2.COLORMAP_PLASMA)
        col  = cv2.resize(col, (W, H))
        frames.append(cv2.cvtColor(col, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def make_comparison_video(depth_frames, base_frames, ctrl_frames, path, fps=16):
    """[Depth | Base WAN | Controlled WAN] side-by-side with labels."""
    T = min(len(depth_frames), len(base_frames), len(ctrl_frames))
    H, W = base_frames.shape[1], base_frames.shape[2]

    def resize_seq(seq):
        return [cv2.resize(f, (W, H)) for f in seq[:T]]

    d_list = resize_seq(depth_frames)
    b_list = resize_seq(base_frames)
    c_list = resize_seq(ctrl_frames)

    label_h  = 30
    canvas_h = H + label_h
    canvas_w = W * 3

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    fthick = 1
    labels = ["Depth (ref video)", "Base WAN (no control)", "Controlled WAN"]
    bgs    = [(40, 40, 40), (20, 60, 20), (80, 20, 20)]  # RGB
    rows = []

    for i in range(T):
        row = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        panels = [d_list[i], b_list[i], c_list[i]]
        for col_idx, (panel, lbl, bg) in enumerate(zip(panels, labels, bgs)):
            x0    = col_idx * W
            strip = np.full((label_h, W, 3), bg, np.uint8)
            (tw, th), _ = cv2.getTextSize(lbl, font, fscale, fthick)
            cv2.putText(strip, lbl,
                        (max(0, (W - tw) // 2), (label_h + th) // 2 - 2),
                        font, fscale, (220, 220, 220), fthick, cv2.LINE_AA)
            row[:label_h, x0:x0 + W] = strip
            row[label_h:, x0:x0 + W] = panel
        rows.append(row)

    save_rgb_video(np.stack(rows), path, fps=fps)
    print(f"    Saved: {path}  ({T} frames, side-by-side)")



def parse_args():
    p = argparse.ArgumentParser(description="ControllableWAN comparison inference")
    p.add_argument('--ref_video',   required=True,
                   help='Reference video — depth extracted from this')
    p.add_argument('--ref_image',   required=True,
                   help='Reference image — first frame for TI2V')
    p.add_argument('--prompt',      required=True)
    p.add_argument('--checkpoint',  required=True,
                   help='checkpoint_*.pt with adapter + zero_convs')
    p.add_argument('--wan_dir',     default='Wan2.2/Wan2.2-TI2V-5B')
    p.add_argument('--output_dir',  default='results')
    p.add_argument('--size',        default='480*832',
                   help='WAN size string e.g. 480*832 or 720*1280')
    p.add_argument('--frame_num',   type=int, default=81,
                   help='Frames to generate — must be 4n+1 (17, 33, 49, 81...)')
    p.add_argument('--steps',       type=int,   default=40)
    p.add_argument('--guidance',    type=float, default=3.0)
    p.add_argument('--fps',         type=int,   default=16)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--depth_hw',    type=int,   nargs=2, default=[128, 128],
                   metavar=('H', 'W'),
                   help='Resolution for depth control tensor — match training')
    p.add_argument('--midas_target_hw', type=int, nargs=2, default=[360, 640],
                   metavar=('H', 'W'),
                   help='Intermediate MiDaS depth size used before raw-depth control resize')
    p.add_argument('--models_dir', default='models',
                   help='Directory containing midas_v3_dpt_large.pth')
    p.add_argument('--midas_weights', default=None,
                   help='Explicit path to midas_v3_dpt_large.pth')
    p.add_argument('--allow_midas_download', action='store_true',
                   help='Fallback to torch.hub pretrained MiDaS if local weights are missing')
    p.add_argument('--offload',     action='store_true', default=True,
                   help='Offload WAN to CPU between steps (saves VRAM)')
    p.add_argument('--no_offload',  dest='offload', action='store_false',
                   help='Keep WAN resident on GPU during generation')
    p.add_argument(
        '--control_strengths',
        type=float,
        nargs='+',
        default=[0.5],
        help=(
            'Residual strengths evaluated using one shared Base WAN run; '
            'default: 0.5'
        ),
    )
    return p.parse_args()



def main():
    args    = parse_args()
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_hw = tuple(args.depth_hw)

    print("\n" + "="*70)
    print("ControllableWAN — Base vs Controlled Comparison")
    print("="*70)
    print(f"  ref_video  : {args.ref_video}")
    print(f"  ref_image  : {args.ref_image}")
    print(f"  prompt     : {args.prompt}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  wan_dir    : {args.wan_dir}")
    print(f"  size       : {args.size}  |  frame_num : {args.frame_num}")
    print(f"  steps      : {args.steps}  |  guidance  : {args.guidance}")
    print(f"  seed       : {args.seed}  |  depth_hw  : {depth_hw}")
    print(f"  midas_hw   : {tuple(args.midas_target_hw)}")
    print(f"  output_dir : {out_dir}")
    print("="*70 + "\n")

    cfg = WAN_CONFIGS['ti2v-5B']

   
    print("[1/4] Extracting depth from reference video...")
    midas, midas_tf = load_midas(
        device=device,
        models_dir=args.models_dir,
        midas_weights=args.midas_weights,
        allow_download=args.allow_midas_download,
    )
    depth_seq = extract_depth_sequence(
        args.ref_video, midas, midas_tf, device,
        num_frames=args.frame_num,
        midas_hw=tuple(args.midas_target_hw),
        control_hw=depth_hw,
    )
    print(f"  depth shape : {depth_seq.shape}  "
          f"min={depth_seq.min():.3f}  max={depth_seq.max():.3f}")
    del midas, midas_tf
    torch.cuda.empty_cache()
    gc.collect()

    print("\n[2/4] Building WanTI2V pipeline...")
    wan_pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.wan_dir,
        device_id=0,
        rank=0,
        t5_cpu=True,
       
    )

    print("  Freeing pipeline vanilla DiT before loading ControllableWAN...")
    wan_pipeline.model.to('cpu')
    del wan_pipeline.model
    wan_pipeline.model = None
    torch.cuda.empty_cache()
    gc.collect()

    print("\n[3/4] Loading ControllableWAN...")
    ctrl_model = ControllableWAN(checkpoint_dir=args.wan_dir, device=device)
    ctrl_model.eval()

    print(f"\n[4/4] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ctrl_model.control_adapter.load_state_dict(ckpt['model'])
    if 'zero_convs' in ckpt:
        ctrl_model.zero_convs.load_state_dict(ckpt['zero_convs'])
        print("  Loaded adapter + zero_convs")
    else:
        print("  WARNING: no zero_convs key in checkpoint")
    print(f"  step={ckpt.get('global_step','?')}  "
          f"best_val_loss={ckpt.get('best_val_loss','?')}")
    del ckpt
    torch.cuda.empty_cache()

    wan_pipeline.model = ctrl_model.wan

    if not args.offload:
        print("  Moving swapped ControllableWAN DiT to GPU...")
        wan_pipeline.model.to(device)
        torch.cuda.empty_cache()

    print("  Pipeline DiT swapped → ControllableWAN.wan (hooks active)")

    ref_image = Image.open(args.ref_image).convert("RGB")

    print("\n" + "-"*60)
    print("Run A — BASE  (ControllableWAN, _control_signal = None)")
    print("-"*60)

    deactivate_adapter(ctrl_model)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        video_base = wan_pipeline.generate(
            args.prompt,
            img=ref_image,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            sampling_steps=args.steps,
            guide_scale=args.guidance,
            seed=args.seed,
            offload_model=args.offload,
        )

    describe_video_tensor("base", video_base)
    frames_base = tensor_to_frames(video_base)
    path_base   = out_dir / "base.mp4"
    save_rgb_video(frames_base, path_base, fps=args.fps)
    save_debug_frames(frames_base, out_dir, "base")
    del video_base
    torch.cuda.empty_cache()
    gc.collect()

    
    print("\n" + "-"*60)
    print("Run B — CONTROLLED  (depth adapter active)")
    print("-"*60)

    control_features = {
        'depth_encoded': depth_to_control_tensor(depth_seq, device)
    }
    print(f"  control tensor : {control_features['depth_encoded'].shape}")

    strengths = [float(value) for value in args.control_strengths]
    if any(value < 0 for value in strengths):
        raise ValueError("Control strengths must be non-negative")

    T_out = frames_base.shape[0]
    depth_for_comparison = depth_seq
    if len(depth_for_comparison) != T_out:
        idxs = np.linspace(
            0, len(depth_for_comparison) - 1, T_out, dtype=int
        )
        depth_for_comparison = depth_for_comparison[idxs]

    depth_rgb = depth_to_rgb(
        depth_for_comparison,
        target_hw=(frames_base.shape[1], frames_base.shape[2]),
    )

    activate_adapter(ctrl_model, control_features)
    result_paths = []

    for strength in strengths:
        tag = f"{strength:g}".replace(".", "p")
        ctrl_model._control_strength = strength

        print("\n" + "-" * 60)
        print(f"CONTROLLED strength={strength:g}")
        print("-" * 60)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        with torch.no_grad():
            video_ctrl = wan_pipeline.generate(
                args.prompt,
                img=ref_image,
                size=SIZE_CONFIGS[args.size],
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                sampling_steps=args.steps,
                guide_scale=args.guidance,
                seed=args.seed,
                offload_model=args.offload,
            )

        describe_video_tensor(
            f"controlled strength={strength:g}", video_ctrl
        )
        frames_ctrl = tensor_to_frames(video_ctrl)

        path_ctrl = out_dir / f"controlled_strength_{tag}.mp4"
        save_rgb_video(frames_ctrl, path_ctrl, fps=args.fps)
        save_debug_frames(
            frames_ctrl,
            out_dir,
            f"controlled_strength_{tag}",
        )

        path_cmp = out_dir / f"comparison_strength_{tag}.mp4"
        make_comparison_video(
            depth_rgb,
            frames_base,
            frames_ctrl,
            path_cmp,
            fps=args.fps,
        )

        result_paths.append((strength, path_ctrl, path_cmp))

        del video_ctrl, frames_ctrl
        torch.cuda.empty_cache()
        gc.collect()

    deactivate_adapter(ctrl_model)

    
    print("\n" + "="*70)
    print("Done")
    print(f"  base.mp4        : {path_base}")
    for strength, controlled_path, comparison_path in result_paths:
        print(f"  strength={strength:g}")
        print(f"    controlled : {controlled_path}")
        print(f"    comparison : {comparison_path}")
    print("="*70)
    print("\nWhat to look for:")
    print("  Depth col    — depth signal fed to adapter (plasma colourmap)")
    print("  Base col     — WAN with prompt only, no control")
    print("  Controlled   — WAN with depth adapter active, same seed")
    print()
    print("  Working  : spatial layout in 'Controlled' tracks depth structure")
    print("  Not yet  : both video cols look identical → adapter has no effect\n")


if __name__ == '__main__':
    main()
