import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Utils

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_video_files(root: Path, exts=(".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def list_frame_folders(root: Path) -> List[Path]:
    # A frame folder is any directory containing at least one image frame
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    folders = []
    for d in [p for p in root.rglob("*") if p.is_dir()]:
        if any((d / f).suffix.lower() in img_exts for f in os.listdir(d)):
            folders.append(d)
    return folders

# Video reading

def read_video_cv2(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def read_frame_folder(path: Path) -> List[np.ndarray]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([p for p in path.iterdir() if p.suffix.lower() in img_exts])
    frames = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return frames

def resize_and_normalize(frames: List[np.ndarray], size: Tuple[int, int]) -> np.ndarray:
    """Resize to (H,W) and scale to [0,1], return shape (T,H,W,C)."""
    H, W = size
    out = []
    for f in frames:
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        out.append(f.astype(np.float32) / 255.0)
    if not out:
        return np.empty((0, H, W, 3), dtype=np.float32)
    return np.stack(out, axis=0)

# Dataset

class ClipDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_mode: str = "video",  # "video" or "frames"
        clip_len: int = 16,
        size: Tuple[int, int] = (128, 192),
        stride: int = 4,
        grayscale: bool = False,
        limit_videos: Optional[int] = None,
    ):
        """
        - root: folder berisi video atau folder-frame
        - data_mode: "video" (cari file .mp4/.avi/...) atau "frames" (tiap video adalah folder frame)
        - clip_len: jumlah frame per klip
        - size: (H, W) target resize
        - stride: langkah slide antar klip (frame)
        - grayscale: jika True, pakai 1 channel
        - limit_videos: batasi jumlah sumber video (debug/CPU training)
        """
        self.root = Path(root)
        self.data_mode = data_mode
        self.clip_len = clip_len
        self.size = size
        self.stride = stride
        self.grayscale = grayscale

        if data_mode == "video":
            self.sources = list_video_files(self.root)
        else:
            self.sources = list_frame_folders(self.root)

        if limit_videos is not None:
            self.sources = self.sources[:limit_videos]

        # Build index of (source_idx, start_frame)
        self.index = []
        for si, src in enumerate(self.sources):
            frames = read_video_cv2(src) if data_mode == "video" else read_frame_folder(src)
            arr = resize_and_normalize(frames, self.size)
            T = arr.shape[0]
            for s in range(0, max(0, T - clip_len + 1), self.stride):
                self.index.append((si, s))
        # Cache to avoid re-reading
        self.cache = {}

    def __len__(self):
        return len(self.index)

    def _get_frames(self, src: Path) -> np.ndarray:
        if src in self.cache:
            return self.cache[src]
        frames = read_video_cv2(src) if self.data_mode == "video" else read_frame_folder(src)
        arr = resize_and_normalize(frames, self.size)
        self.cache[src] = arr
        return arr

    def __getitem__(self, idx):
        si, s = self.index[idx]
        src = self.sources[si]
        arr = self._get_frames(src)  # (T,H,W,C)
        clip = arr[s : s + self.clip_len]
        if self.grayscale and clip.shape[-1] == 3:
            clip = np.mean(clip, axis=-1, keepdims=True)
        # (T,H,W,C) -> (C,T,H,W) tensor
        clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)
        return torch.from_numpy(clip), str(src), s

# Model: 3D Conv Autoencoder

class Conv3dAutoEncoder(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # Encoder: reduce spatial+temporal
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(base), nn.ReLU(inplace=True),
            nn.Conv3d(base, base*2, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(base*2), nn.ReLU(inplace=True),
            nn.Conv3d(base*2, base*4, kernel_size=3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(base*4), nn.ReLU(inplace=True),
            nn.Conv3d(base*4, base*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base*4), nn.ReLU(inplace=True),
        )
        # Decoder: mirror
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(base*4, base*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base*4), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base*4, base*2, kernel_size=4, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(base*2), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base*2, base, kernel_size=4, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(base), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base, in_ch, kernel_size=4, stride=(1,2,2), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat

# Training & Evaluation

def build_loaders(args):
    train_set = ClipDataset(
        root=args.data_root,
        data_mode=args.data_mode,
        clip_len=args.clip_len,
        size=(args.height, args.width),
        stride=args.train_stride,
        grayscale=args.grayscale,
        limit_videos=args.limit_videos,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    return train_loader


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train(args):
    device = torch.device("cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)

    loader = build_loaders(args)
    in_ch = 1 if args.grayscale else 3
    model = Conv3dAutoEncoder(in_ch=in_ch, base=args.base_channels).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))
    crit = nn.MSELoss()

    best_loss = float('inf')
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / "checkpoints"

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{args.epochs}")
        for clips, _, _ in pbar:
            clips = clips.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.amp)):
                preds = model(clips)
                # Samakan panjang temporal output dan target
            if preds.shape[2] != clips.shape[2]:
                min_len = min(preds.shape[2], clips.shape[2])
                preds = preds[:, :, :min_len]
                clips = clips[:, :, :min_len]

            loss = crit(preds, clips)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * clips.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running / len(loader.dataset)
        print(f"epoch_loss={epoch_loss:.6f}")
        # save last
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'args': vars(args),
        }, ckpt_dir / 'last.pt')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'args': vars(args),
                'best_loss': best_loss,
            }, ckpt_dir / 'best.pt')
            print(f"✓ saved new best (loss={best_loss:.6f})")


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)

    # Build dataset without shuffling; stride defines density of clips
    ds = ClipDataset(
        root=args.data_root,
        data_mode=args.data_mode,
        clip_len=args.clip_len,
        size=(args.height, args.width),
        stride=args.eval_stride,
        grayscale=args.grayscale,
        limit_videos=args.limit_videos,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True)

    in_ch = 1 if args.grayscale else 3
    model = Conv3dAutoEncoder(in_ch=in_ch, base=args.base_channels).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    crit = nn.MSELoss(reduction='none')

    # We'll compute per-clip MSE, then min-max normalize into regularity score: 1 - norm_err
    clip_errors = []  # (video_path, start_idx, mse)

    for clips, paths, starts in tqdm(dl, total=len(dl), desc="Evaluating"):
        clips = clips.to(device, non_blocking=True)
        preds = model(clips)
        # Samakan jumlah frame (dimensi temporal) sebelum hitung loss
        if preds.shape[2] != clips.shape[2]:
            min_len = min(preds.shape[2], clips.shape[2])
            preds = preds[:, :, :min_len]
            clips = clips[:, :, :min_len]

        mse = crit(preds, clips)

        # reduce over channels, time, H, W
        mse = mse.mean(dim=(1,2,3,4)).detach().cpu().numpy()
        for p, s, e in zip(paths, starts.tolist(), mse.tolist()):
            clip_errors.append((p, s, e))

    # Normalize to [0,1] (higher means more regular like in the paper)
    errs = np.array([e for _,_,e in clip_errors], dtype=np.float64)
    if len(errs) == 0:
        print("No clips, nothing to export.")
        return
    mn, mx = float(errs.min()), float(errs.max())
    denom = (mx - mn) if mx > mn else 1.0
    scores = 1.0 - ((errs - mn) / denom)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["video", "clip_start_frame", "mse", "regularity_score"])
        for (p, s, e), sc in zip(clip_errors, scores):
            w.writerow([p, s, f"{e:.8f}", f"{sc:.8f}"])
    print(f"✓ Wrote scores to: {out_csv}")

# CLI

def build_argparser():
    p = argparse.ArgumentParser(description="Temporal Regularity Autoencoder (PyTorch)")
    p.add_argument('--mode', choices=['train', 'eval'], required=True)
    p.add_argument('--data_root', type=str, required=True, help='Root of videos or frame-folders')
    p.add_argument('--data_mode', choices=['video', 'frames'], default='video')
    p.add_argument('--clip_len', type=int, default=16)
    p.add_argument('--height', type=int, default=128)
    p.add_argument('--width', type=int, default=192)
    p.add_argument('--train_stride', type=int, default=4)
    p.add_argument('--eval_stride', type=int, default=1)
    p.add_argument('--grayscale', action='store_true', help='Use 1-channel input')
    p.add_argument('--limit_videos', type=int, default=None, help='Debug: limit number of sources')

    # Optimization
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--base_channels', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='Enable mixed precision on CUDA')
    p.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    p.add_argument('--seed', type=int, default=42)

    # Paths
    p.add_argument('--work_dir', type=str, default='./runs/default')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--out_csv', type=str, default='./runs/default/scores.csv')

    return p

def main():
    args = build_argparser().parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            print("--checkpoint is required for eval and must exist", file=sys.stderr)
            sys.exit(2)
        evaluate(args)
    else:
        raise ValueError(args.mode)


if __name__ == '__main__':
    main()
