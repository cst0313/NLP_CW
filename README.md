# SemEval-2022 Task 4 — Severity-Aware Soft-Label DeBERTa

**Leaderboard name:** cst0313  
**Dev macro-F1:** 0.7854  
**Threshold:** 0.35  

## Repository Structure
```
BestModel/      Training and prediction scripts
eda/            EDA notebook (produces all EDA figures)
analysis/       Error analysis, calibration, PR curve notebooks
figures/        All figures referenced in the report
dev.txt         Dev set predictions (one per line, 0 or 1)
test.txt        Test set predictions (one per line, 0 or 1)
```

## Setup

Python 3.9+ required.
```bash
pip install -r requirements.txt
```

## Data

Download the dataset from the official task page:  
https://github.com/Perez-AlmendrosC/dontpatronizeme

Place `dontpatronizeme_pcl.tsv` in the repo root before running any scripts.

## Reproducing Training
```bash
python BestModel/train.py \
  --data_path dontpatronizeme_pcl.tsv \
  --output_dir BestModel/checkpoint \
  --epochs 10 \
  --threshold 0.35 \
  --seed 42
```

Expected dev macro-F1: **0.7854** at threshold 0.35.

## Generating Predictions
```bash
# Dev set
python BestModel/predict.py \
  --data_path dontpatronizeme_pcl.tsv \
  --split_ids data/splits/dev_semeval_parids-labels.csv \
  --model_path BestModel/checkpoint/best_model_pcl.pt \
  --output dev.txt \
  --threshold 0.35

# Test set  
python BestModel/predict.py \
  --data_path data/raw/task4_test.tsv \
  --model_path BestModel/checkpoint/best_model_pcl.pt \
  --output test.txt \
  --threshold 0.35
```

## Model Details

- **Encoder:** microsoft/deberta-v3-base
- **Soft label targets:** {0→0.0, 1→0.1, 2→0.7, 3→0.9, 4→1.0}
- **Loss:** Focal Soft-BCE (γ=2)
- **Imbalance handling:** WeightedRandomSampler
- **Learning rate:** AdamW with LLRD (α=0.9, base lr=2e-5)
- **Max sequence length:** 256 tokens
- **Epochs:** 10 with early stopping on dev macro-F1
- **Classification threshold:** 0.35 (dev-optimised)

## Note on Model Weights

Model weights are not included in this repository due to file size.  
Running train.py with seed 777 should reproduce the reported results.
