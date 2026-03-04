"""Video folder dataset for TimeSformer experiments.

Expected layout:
  dataset_root/
    class_a/
    class_b/

Each video is decoded on the fly and converted to a normalized float tensor in
__getitem__:
  - sample a clip index sequence
  - resize to target HxW
  - normalize with ImageNet mean/std

No preprocessed video files are required.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_frames: int,
        image_size: int,
        extensions: Sequence[str],
        class_to_idx: Optional[Dict[str, int]] = None,
        random_clip: bool = False,
        sampling_rate: int = 1,
    ) -> None:
        root_path = Path(root_dir)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.data_dir = self._resolve_data_dir(root_path)
        self.class_to_idx = {str(name): int(idx) for name, idx in dict(class_to_idx or {}).items()}
        self.random_clip = bool(random_clip)
        self.sampling_rate = max(1, int(sampling_rate))

        assert self.num_frames > 0, "num_frames must be positive."
        assert self.image_size > 0, "image_size must be positive."
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        if not self.class_to_idx:
            self._build_class_map()
        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No video files found in: {self.data_dir}")

    def _iter_class_dirs(self) -> List[Path]:
        return sorted([p for p in self.data_dir.iterdir() if p.is_dir()])

    def _resolve_data_dir(self, root_path: Path) -> Path:
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")
        return root_path

    def _build_class_map(self) -> None:
        class_dirs = self._iter_class_dirs()
        if not class_dirs:
            raise RuntimeError(f"No class subdirectories in: {self.data_dir}")

        self.class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}

    def _validate_class_map(self, class_dir_names: List[str]) -> None:
        if not self.class_to_idx:
            return

        provided = set(self.class_to_idx.keys())
        available = set(class_dir_names)
        missing = sorted(available - provided)
        unknown = sorted(provided - available)
        if missing:
            raise ValueError(
                f"Class directories not in provided class_to_idx for {self.data_dir}: {', '.join(missing)}"
            )
        if unknown:
            raise ValueError(
                f"class_to_idx contains unknown classes for {self.data_dir}: {', '.join(unknown)}"
            )

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        class_dirs = self._iter_class_dirs()
        class_dir_names = [p.name for p in class_dirs]

        if self.class_to_idx:
            self._validate_class_map(class_dir_names)
        else:
            self._build_class_map()
            self._validate_class_map(class_dir_names)

        class_name_to_idx = {name: self.class_to_idx[name] for name in class_dir_names}
        collected: List[Tuple[Path, int]] = []
        for class_name in class_dir_names:
            class_dir = self.data_dir / class_name
            for file_path in sorted(class_dir.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                    collected.append((file_path, class_name_to_idx[class_name]))

        return collected

    def _sample_indices(self, total_frames: int) -> torch.Tensor:
        available = torch.arange(0, total_frames, step=self.sampling_rate)
        if available.numel() < self.num_frames:
            raise RuntimeError(
                f"Not enough frames after sampling. total={total_frames}, sampling_rate={self.sampling_rate}, "
                f"required={self.num_frames}, got={available.numel()}"
            )

        if self.random_clip:
            max_start = available.numel() - self.num_frames
            start = torch.randint(0, max_start + 1, (1,)).item()
            return available[start : start + self.num_frames]

        positions = torch.linspace(0, available.numel() - 1, steps=self.num_frames).long()
        return available[positions]

    def _preprocess(self, video: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        clip = video[indices].float().div(255.0).permute(0, 3, 1, 2).contiguous()
        clip = F.interpolate(
            clip,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype).view(1, 3, 1, 1)
        return (clip - mean) / std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
        video, _, _ = read_video(str(video_path), pts_unit="sec")
        if video.numel() == 0:
            raise RuntimeError(f"Failed to decode video: {video_path}")
        if video.dim() != 4 or video.size(-1) != 3:
            raise RuntimeError(f"Unexpected video shape: {video.shape}")

        indices = self._sample_indices(video.shape[0])
        video = self._preprocess(video, indices)

        return {
            "pixel_values": video,
            "labels": torch.tensor(label, dtype=torch.long),
            "path": str(video_path),
        }


def collate_video_batch(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "paths": [item["path"] for item in batch],
    }
