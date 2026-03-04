# TimeSformer Evaluation Summary (UCF101 + HMDB51)

## Experiment Overview
- **Base model**: `facebook/timesformer-base-finetuned-k400`

## Overall Comparison
| Dataset | #Samples | #Classes | Top-1 Acc | Top-5 Acc | Macro Precision | Macro Recall | Macro F1 | Micro Precision | Micro Recall | Micro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UCF101 | 1332 | 101 | 91.97% | 98.35% | 92.88% | 91.73% | 91.72% | 91.97% | 91.97% | 91.97% | 92.94% | 91.97% | 91.89% |
| HMDB51 | 677 | 51 | 63.52% | 85.23% | 71.73% | 66.61% | 66.98% | 63.52% | 63.52% | 63.52% | 67.49% | 63.52% | 63.34% |

> Note: For single-label multi-class classification, **Micro F1 equals Top-1 Accuracy**.

## Artifacts
- UCF101 confusion heatmap: `outputs/run_dir/ucf101_confusion_matrix.png`
- HMDB51 confusion heatmap: `outputs/run_dir/hmdb51_confusion_matrix.png`

---

# TimeSformer Inference Skeleton (Python Module)

This project runs fine-tuning (optional) and inference with a pre-trained TimeSformer model.
It keeps the notebook workflow in mind, so you can run everything from Python
modules in Jupyter as well as from CLI.

## Folder Layout

```text
timesformer/
  configs/
    base.yaml
  src/
    data/
      video_folder_dataset.py
    models/
      timesformer_experiment.py
    train.py
  requirements.txt
```

## Data Layout

```text
src/data/
  class_a/
    clip_0001.mp4
  class_b/
    clip_0002.mp4
```

Each class folder is treated as one label.  
By default the data is split into train/val/test with ratio 8:1:1 using stratified `train_test_split`.

## Run

### CLI

```bash
python -m src.train --config configs/base.yaml
```

### Jupyter

```jupyter
project_root = Path.cwd().parent
config_path = project_root / "configs" / "base.yaml"
results = infer_from_config(config_path)
print(results)
```

## What it does

1. Loads a Hugging Face TimeSformer checkpoint (`model.pretrained_name`).
2. Optionally swaps the classifier head (`model.use_custom_head`).
3. Decodes videos in folder dataset and converts them to model input tensors.
4. Runs validation during training (optional), then runs final inference on the test split and saves:
   - overall metrics (top-1/top-5 etc.)
   - confusion matrix and per-class accuracy
   - per-sample predictions CSV

## Example config

```yaml
seed: 42
device: "cuda"

data:
  train_dir: "./src/data"
  eval_dir: "./src/data"
  extensions: [".mp4", ".avi", ".mov", ".mkv"]
  num_frames: 8
  image_size: 224
  batch_size: 4
  num_workers: 0
  random_clip_eval: true
  sampling_rate: 1

model:
  pretrained_name: "facebook/timesformer-base-finetuned-k400"
  num_classes: null
  dropout: 0.1
  attention_dropout: 0.1
  freeze_backbone: false
  use_custom_head: false
  custom_head_hidden_dim: 768
  # checkpoint: "./outputs/..."  # optional: load a custom checkpoint

inference:
  run_id: null
  output_dir: "./outputs"
  topk: [1, 5]
  write_predictions: true

training:
  enabled: true
  epochs: 15
  lr: 0.0001
  weight_decay: 0.0
  grad_clip_norm: 1.0
  mixed_precision: true
  val_ratio: 0.1
  test_ratio: 0.1
  val_every: 1
```

## Outputs

Artifacts are written under `outputs/<run_id>/`:

- `results.json` - summary metrics
- `predictions.csv` - sample-wise predictions and scores
- `config.yaml` - config snapshot
- `run_meta.json` - run metadata

## Optional finetune before inference

To adapt the random-initialized custom head, set:

```yaml
training:
  enabled: true
  epochs: 15
```

This runs the configured `epochs` passes before final inference.
