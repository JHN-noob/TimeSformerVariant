import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.video_folder_dataset import VideoFolderDataset, collate_video_batch
from src.models.timesformer_experiment import TimeSformerExperiment


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _build_dataset(
    root: Path,
    data_cfg: Dict,
    random_clip: bool,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> VideoFolderDataset:
    return VideoFolderDataset(
        root_dir=str(root),
        num_frames=int(data_cfg["num_frames"]),
        image_size=int(data_cfg["image_size"]),
        extensions=data_cfg["extensions"],
        class_to_idx=class_to_idx,
        random_clip=random_clip,
        sampling_rate=int(data_cfg.get("sampling_rate", 1)),
    )


def _build_loader(dataset: VideoFolderDataset, data_cfg: Dict, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_video_batch,
    )


class _SplitDataset(Dataset):
    def __init__(self, base_dataset: VideoFolderDataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_to_idx = base_dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


def build_dataloaders(cfg: Dict, project_root: Path) -> Tuple[VideoFolderDataset, Optional[DataLoader]]:
    data_cfg = cfg["data"]
    if "train_dir" not in data_cfg and "eval_dir" not in data_cfg:
        raise KeyError("config.data.train_dir or config.data.eval_dir is required.")

    source_root = resolve_path(project_root, data_cfg.get("train_dir", data_cfg["eval_dir"]))
    source_dataset = _build_dataset(
        source_root,
        data_cfg,
        random_clip=bool(data_cfg.get("random_clip_eval", True)),
        class_to_idx=None,
    )

    _, _, test_loader = build_train_dataloaders(cfg, project_root, class_to_idx=source_dataset.class_to_idx)
    if test_loader is None:
        test_loader = _build_loader(source_dataset, data_cfg, shuffle=False)
    return source_dataset, test_loader


def _safe_split(
    values: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
    require_stratified: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if test_size <= 0:
        return values, np.array([], dtype=int)
    if test_size >= 1:
        raise ValueError("split size must be < 1.")
    if not require_stratified or len(np.unique(labels)) == 1:
        return train_test_split(values, test_size=test_size, random_state=seed, shuffle=True)

    try:
        return train_test_split(
            values,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
            shuffle=True,
        )
    except ValueError:
        return train_test_split(values, test_size=test_size, random_state=seed, shuffle=True)


def _split_indices(
    labels: List[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if not (0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("training.val_ratio and training.test_ratio must be in [0, 1].")
    if val_ratio + test_ratio >= 1:
        raise ValueError("training.val_ratio + training.test_ratio must be < 1.")

    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("dataset is empty.")

    indices = np.arange(n_samples)
    labels_arr = np.array(labels)
    valtest_ratio = val_ratio + test_ratio
    if valtest_ratio <= 0:
        return indices.tolist(), [], []

    train_idx, valtest_idx = _safe_split(
        indices,
        labels_arr,
        test_size=valtest_ratio,
        seed=seed,
        require_stratified=True,
    )

    if val_ratio == 0:
        return train_idx.tolist(), [], valtest_idx.tolist()
    if test_ratio == 0:
        return train_idx.tolist(), valtest_idx.tolist(), []

    rel_test_ratio = test_ratio / valtest_ratio
    val_idx, test_idx = _safe_split(
        np.array(valtest_idx),
        labels_arr[valtest_idx],
        test_size=rel_test_ratio,
        seed=seed,
        require_stratified=True,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def build_train_dataloaders(
    cfg: Dict,
    project_root: Path,
    class_to_idx: Optional[Dict[str, int]],
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})

    train_root = resolve_path(project_root, data_cfg.get("train_dir", data_cfg["eval_dir"]))
    train_clip = bool(data_cfg.get("random_clip_train", data_cfg.get("random_clip_eval", True)))
    eval_clip = bool(data_cfg.get("random_clip_eval", True))

    train_dataset = _build_dataset(
        train_root,
        data_cfg,
        random_clip=train_clip,
        class_to_idx=class_to_idx,
    )
    eval_dataset = _build_dataset(
        train_root,
        data_cfg,
        random_clip=eval_clip,
        class_to_idx=train_dataset.class_to_idx,
    )

    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    test_ratio = float(train_cfg.get("test_ratio", 0.1))
    train_ratio = 1.0 - val_ratio - test_ratio
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("training.val_ratio and training.test_ratio must be >= 0.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("training.val_ratio + training.test_ratio must be < 1.")

    if train_ratio <= 0:
        raise ValueError("training.val_ratio + training.test_ratio must be < 1.0.")

    all_labels = [label for _, label in train_dataset.samples]
    train_idx, val_idx, test_idx = _split_indices(
        all_labels,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=int(train_cfg.get("split_seed", cfg.get("seed", 42))),
    )

    train_split = _SplitDataset(train_dataset, train_idx)
    val_split = _SplitDataset(eval_dataset, val_idx)
    test_split = _SplitDataset(eval_dataset, test_idx)
    val_loader = _build_loader(val_split, data_cfg, shuffle=False) if val_idx else None
    test_loader = _build_loader(test_split, data_cfg, shuffle=False) if test_idx else None

    return (
        _build_loader(train_split, data_cfg, shuffle=True),
        val_loader,
        test_loader,
    )


def _topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    max_k = min(int(k), logits.size(-1))
    if max_k <= 0:
        return 0
    topk = logits.topk(max_k, dim=-1).indices
    return topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()


def _ensure_topk(topk: Sequence[int] | int) -> Tuple[int, ...]:
    if isinstance(topk, int):
        values = [int(topk)]
    else:
        values = list(topk)

    deduped = sorted({int(v) for v in values if int(v) > 0})
    if not deduped:
        raise ValueError("inference.topk must contain at least one integer >= 1.")
    return tuple(deduped)


def _load_checkpoint(model: nn.Module, project_root: Path, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        return

    checkpoint_file = resolve_path(project_root, checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")

    state = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[checkpoint] loaded with missing={len(missing)} unexpected={len(unexpected)} keys.")


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    topk: Tuple[int, ...],
    run_id: str,
    use_amp: bool,
    val_every: int = 1,
) -> Dict[str, float]:
    print(f"[train] start epochs={epochs}, batches_per_epoch={len(train_loader)}, dataset_size={len(train_loader.dataset)}")
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Enable trainable layers before fine-tuning.")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    topk_values = sorted(topk)
    topk_values = tuple(v for v in topk_values if v > 0)
    if not topk_values:
        topk_values = (1,)

    total_loss = 0.0
    total_seen = 0
    total_correct = {int(k): 0 for k in topk_values}
    best_val = -1.0
    best_val_epoch = 0
    best_val_metrics: Dict[str, float] = {}
    val_every = max(int(val_every), 1)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_seen = 0
        epoch_correct = {int(k): 0 for k in topk_values}

        model.train()
        for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch}/{epochs}", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            epoch_seen += bs
            epoch_loss += loss.item() * bs
            for k in topk_values:
                epoch_correct[k] += _topk_correct(logits, labels, k)

        epoch_loss = epoch_loss / max(epoch_seen, 1)
        epoch_acc = {k: epoch_correct[k] / max(epoch_seen, 1) for k in topk_values}
        for k in topk_values:
            total_correct[k] += epoch_correct[k]
        total_loss += epoch_loss * epoch_seen
        total_seen += epoch_seen

        val_out = ""
        val_metrics = None
        if val_loader is not None and epoch % val_every == 0:
            val_metrics = _evaluate_metrics(model, val_loader, device, topk)
            val_out = (
                f" val_loss={val_metrics.get('loss', 0.0):.4f} "
                f"val_top1={val_metrics.get('acc_top1', 0.0):.4f}"
            )
            if val_metrics["acc_top1"] > best_val:
                best_val = val_metrics["acc_top1"]
                best_val_epoch = epoch
                best_val_metrics = val_metrics

        if epoch == epochs:
            last_epoch_loss = epoch_loss
            last_epoch_acc = epoch_acc
        print(
            f"[{run_id}] epoch={epoch}/{epochs} "
            f"train_loss={epoch_loss:.4f} "
            f"train_acc_top1={epoch_acc.get(1, 0.0):.4f}"
            + val_out
        )

    denom = max(total_seen, 1)
    metrics = {
        "train_loss": total_loss / denom,
    }
    for k in topk_values:
        metrics[f"train_acc_top{k}"] = total_correct[k] / denom
    metrics["epoch"] = epochs
    metrics["last_loss"] = float(last_epoch_loss)
    metrics["last_acc_top1"] = float(last_epoch_acc.get(1, 0.0))
    if val_metrics is not None:
        metrics["val_loss"] = float(val_metrics.get("loss", 0.0))
        metrics["val_acc_top1"] = float(val_metrics.get("acc_top1", 0.0))
    if best_val_metrics:
        metrics["best_val_loss"] = float(best_val_metrics.get("loss", 0.0))
        metrics["best_val_acc_top1"] = float(best_val_metrics.get("acc_top1", 0.0))
        metrics["best_val_epoch"] = int(best_val_epoch)

    return metrics


def _evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: Tuple[int, ...],
) -> Dict[str, float]:
    topk_values = sorted(topk)
    criterion = nn.CrossEntropyLoss()
    num_classes = len(loader.dataset.class_to_idx)
    if num_classes <= 0:
        raise ValueError("Dataset has no classes.")

    total_seen = 0
    total_loss = 0.0
    total_correct = {int(k): 0 for k in topk_values}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validation", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(pixel_values=pixel_values).logits
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total_seen += labels.size(0)
            for k in topk_values:
                total_correct[k] += _topk_correct(logits, labels, k)

    denom = max(total_seen, 1)
    return {
        "num_samples": int(total_seen),
        "num_classes": num_classes,
        "loss": total_loss / denom,
        **{f"acc_top{k}": total_correct[k] / denom for k in topk_values},
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: Tuple[int, ...],
) -> Tuple[Dict, List[Dict[str, str]]]:
    model.eval()
    topk_values = sorted(topk)
    num_classes = len(loader.dataset.class_to_idx)
    max_k = min(max(topk_values), num_classes)
    if max_k <= 0:
        raise ValueError("num_classes must be at least 1.")
    idx_to_class = {idx: name for name, idx in loader.dataset.class_to_idx.items()}

    total_seen = 0
    total_correct = {k: 0 for k in topk_values}
    per_class_total = [0 for _ in range(num_classes)]
    per_class_correct = [0 for _ in range(num_classes)]
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    rows: List[Dict[str, str]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(pixel_values=pixel_values).logits
            probs = F.softmax(logits, dim=-1)
            topk_indices = probs.topk(max_k, dim=-1).indices

            total_seen += labels.size(0)
            for k in topk_values:
                total_correct[k] += _topk_correct(logits, labels, k)

            gt_ids = labels.tolist()
            preds = topk_indices[:, 0].tolist()
            topk_ids = topk_indices.tolist()
            topk_probs = torch.gather(
                probs, 1, topk_indices
            ).tolist()

            for i in range(labels.size(0)):
                gt_id = int(gt_ids[i])
                pred_id = int(preds[i])
                path = batch["paths"][i]
                row_ids = [str(v) for v in topk_ids[i]]
                row_probs = [float(v) for v in topk_probs[i]]

                rows.append(
                    {
                        "path": path,
                        "label_id": str(gt_id),
                        "label_name": idx_to_class.get(gt_id, str(gt_id)),
                        "pred_id": str(pred_id),
                        "pred_name": idx_to_class.get(pred_id, str(pred_id)),
                        "correct": str(int(gt_id == pred_id)),
                        "top1_prob": f"{row_probs[0]:.6f}",
                        "topk_ids": "|".join(row_ids),
                        "topk_names": "|".join(idx_to_class.get(int(v), str(int(v))) for v in row_ids),
                        "topk_probs": "|".join(f"{v:.6f}" for v in row_probs),
                    }
                )

                if 0 <= gt_id < num_classes and 0 <= pred_id < num_classes:
                    confusion[gt_id][pred_id] += 1
                    per_class_total[gt_id] += 1
                    if gt_id == pred_id:
                        per_class_correct[gt_id] += 1

    denom = max(total_seen, 1)
    metrics = {
        "num_samples": int(total_seen),
        "num_classes": num_classes,
    }
    for k in topk_values:
        metrics[f"acc_top{k}"] = total_correct[k] / denom

    metrics["class_accuracy"] = {
        idx_to_class[idx]: (per_class_correct[idx] / per_class_total[idx] if per_class_total[idx] else 0.0)
        for idx in range(num_classes)
    }
    metrics["confusion_matrix"] = confusion

    return metrics, rows


def _build_run_id(cfg: Dict) -> str:
    configured = cfg.get("inference", {}).get("run_id")
    if isinstance(configured, str):
        configured = configured.strip()
        if configured and configured.lower() != "none":
            return configured
    if configured is None:
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if str(configured).strip().lower() == "none":
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return str(configured)


def _compact_metrics(metrics: Dict) -> Dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"class_accuracy", "confusion_matrix"}
    }


def _write_predictions_csv(path: Path, run_id: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "run_id",
        "path",
        "label_id",
        "label_name",
        "pred_id",
        "pred_name",
        "correct",
        "top1_prob",
        "topk_ids",
        "topk_names",
        "topk_probs",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"run_id": run_id, **row})


def _build_model(cfg: Dict, num_labels: int) -> TimeSformerExperiment:
    model_cfg = cfg["model"]
    return TimeSformerExperiment(
        pretrained_name=model_cfg["pretrained_name"],
        num_frames=int(cfg["data"]["num_frames"]),
        image_size=int(cfg["data"]["image_size"]),
        num_labels=num_labels,
        dropout=float(model_cfg["dropout"]),
        attention_dropout=float(model_cfg["attention_dropout"]),
        freeze_backbone=bool(model_cfg["freeze_backbone"]),
        use_custom_head=bool(model_cfg["use_custom_head"]),
        custom_head_hidden_dim=int(model_cfg["custom_head_hidden_dim"]),
    )


def infer_from_config(config_path: str) -> Dict:
    config_path_obj = Path(config_path).resolve()
    project_root = config_path_obj.parent.parent
    cfg = load_config(str(config_path_obj))

    set_seed(int(cfg.get("seed", 42)))

    preferred_device = str(cfg.get("device", "cuda")).lower()
    use_cuda = preferred_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data_cfg = cfg["data"]
    source_root = resolve_path(project_root, data_cfg.get("train_dir", data_cfg["eval_dir"]))
    dataset = _build_dataset(
        source_root,
        data_cfg,
        random_clip=bool(data_cfg.get("random_clip_eval", True)),
        class_to_idx=None,
    )
    train_loader, val_loader, test_loader = build_train_dataloaders(
        cfg,
        project_root,
        class_to_idx=dataset.class_to_idx,
    )
    num_labels = cfg["model"].get("num_classes") or len(dataset.class_to_idx)
    model = _build_model(cfg, int(num_labels)).to(device)

    model_cfg = cfg["model"]
    _load_checkpoint(model, project_root, model_cfg.get("checkpoint"))

    inf_cfg = cfg.get("inference", {})
    topk = _ensure_topk(inf_cfg.get("topk", (1, 5)))
    write_predictions = bool(inf_cfg.get("write_predictions", True))
    configured_output_dir = inf_cfg.get("output_dir", "./outputs")
    if configured_output_dir is None or str(configured_output_dir).strip().lower() in {"", "none", "null"}:
        output_root = project_root / "outputs"
    else:
        output_root = resolve_path(project_root, str(configured_output_dir))
    train_cfg = cfg.get("training", {})
    print(
        f"[config] training.enabled={bool(train_cfg.get('enabled', False))}, "
        f"training.epochs={train_cfg.get('epochs', 15)}"
    )
    train_metrics: Dict = {}

    run_id = _build_run_id(cfg)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "results.json"
    predictions_path = run_dir / "predictions.csv"
    config_snapshot_path = run_dir / "config.yaml"
    run_meta_path = run_dir / "run_meta.json"

    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    if bool(train_cfg.get("enabled", False)):
        train_loader_size = len(train_loader.dataset)
        val_loader_size = len(val_loader.dataset) if val_loader is not None else 0
        test_loader_size = len(test_loader.dataset) if test_loader is not None else 0
        print(
            f"[split] total={len(dataset)} train={train_loader_size} "
            f"val={val_loader_size} test={test_loader_size}"
        )
        if train_loader_size <= 0:
            raise RuntimeError("training split has 0 samples. Check val_ratio/test_ratio values.")

        train_epochs = int(train_cfg.get("epochs", 15))
        train_lr = float(train_cfg.get("lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        mixed_precision = bool(train_cfg.get("mixed_precision", True))
        val_every = int(train_cfg.get("val_every", 1))

        train_metrics = _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=train_epochs,
            lr=train_lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            topk=topk,
            run_id=run_id,
            use_amp=mixed_precision and device.type == "cuda",
            val_every=val_every,
        )

        finetuned_ckpt_path = run_dir / "finetuned.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "class_to_idx": dataset.class_to_idx, "run_id": run_id},
            finetuned_ckpt_path,
        )
        if "finetuned_ckpt" not in train_metrics:
            train_metrics["finetuned_ckpt"] = str(finetuned_ckpt_path)

    eval_loader = test_loader
    if eval_loader is None:
        eval_loader = val_loader if val_loader is not None else train_loader

    metrics, rows = _evaluate(model, eval_loader, device, topk)
    if write_predictions:
        _write_predictions_csv(predictions_path, run_id, rows)

    summary = {
        "run_id": run_id,
        "model_name": model_cfg["pretrained_name"],
        "topk": topk,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "run_dir": str(run_dir),
        "eval_dir": str(resolve_path(project_root, cfg["data"].get("train_dir", cfg["data"]["eval_dir"]))),
    }

    run_meta = {
        "run_id": run_id,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(cfg.get("seed", 42)),
        "device": str(device),
        "class_to_idx": dataset.class_to_idx,
        "num_classes": len(dataset.class_to_idx),
        "config_path": str(config_path_obj),
        "run_dir": str(run_dir),
        "results_path": str(summary_path),
        "predictions_path": str(predictions_path) if write_predictions else None,
        "config_snapshot": str(config_snapshot_path),
    }
    if bool(train_cfg.get("enabled", False)):
        run_meta.update(train_metrics)
    run_meta.update(_compact_metrics(summary["metrics"]))

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    printed_summary = _compact_metrics(summary["metrics"])
    print(
        f"[{run_id}] samples={summary['metrics']['num_samples']} "
        f"acc_top1={summary['metrics']['acc_top1']:.4f} "
        f"acc_top5={summary['metrics'].get('acc_top5', 0.0):.4f}"
    )
    print(f"summary: {summary_path}")
    if write_predictions:
        print(f"predictions: {predictions_path}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": printed_summary,
        "results_path": str(summary_path),
        "run_meta_path": str(run_meta_path),
        "predictions_path": str(predictions_path) if write_predictions else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    results = infer_from_config(args.config)
    print(results)


if __name__ == "__main__":
    main()


'''if you use a jupyter notebook
project_root = Path.cwd().parent
config_path = project_root / "configs" / "base.yaml"
results = infer_from_config(config_path)
print(results)
'''
