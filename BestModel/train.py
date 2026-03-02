# train.py
# ─────────────────────────────────────────────────────────────────────────────
# Trains DeBERTa-v3-base with severity-aware soft labels and focal loss
# for SemEval-2022 Task 4 Subtask 1 (Patronizing and Condescending Language).
#
# Expected output: saves best model checkpoint to --output_dir
# Expected dev macro-F1: ~0.7854 at threshold 0.35 (seed-dependent)
#
# Usage:
#   python train.py \
#     --data_path dontpatronizeme_pcl.tsv \
#     --train_ids data/splits/train_semeval_parids-labels.csv \
#     --dev_ids   data/splits/dev_semeval_parids-labels.csv \
#     --output_dir BestModel/checkpoint \
#     --epochs 10 \
#     --threshold 0.35 \
#     --seed 42
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import random
import logging
from collections import Counter
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

logging.basicConfig(level=logging.WARNING)


# ── Constants (also configurable via argparse) ─────────────────────────────────
MODEL_NAME  = "microsoft/deberta-v3-base"
MAX_LENGTH  = 256
BATCH_SIZE  = 8
NUM_EPOCHS  = 10
BASE_LR     = 2e-5
HEAD_LR     = 1e-4
LR_DECAY    = 0.9    # LLRD decay factor
WARMUP_FRAC = 0.1
FOCAL_GAMMA = 2.0
DROPOUT     = 0.1

# Soft label mapping — maps original 0-4 severity to soft probability targets
# Motivation: EDA shows label=1 examples have 26% error rate and near-0.5
# predicted probabilities from a TF-IDF baseline — they are genuinely uncertain.
# Discarding this ambiguity gradient wastes a training signal preserved in
# the original annotations. See paper Section 3 for full EDA justification.
SOFT_TARGET_MAP = {0: 0.0, 1: 0.1, 2: 0.7, 3: 0.9, 4: 1.0}


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train severity-aware soft-label DeBERTa for PCL detection"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to dontpatronizeme_pcl.tsv"
    )
    parser.add_argument(
        "--train_ids", type=str,
        default="data/splits/train_semeval_parids-labels.csv",
        help="CSV with par_id column for training split"
    )
    parser.add_argument(
        "--dev_ids", type=str,
        default="data/splits/dev_semeval_parids-labels.csv",
        help="CSV with par_id column for dev split"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoint",
        help="Directory to save the best model checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BASE_LR,
                        help="Base (encoder) learning rate for LLRD")
    parser.add_argument("--head_lr", type=float, default=HEAD_LR,
                        help="Classification head learning rate")
    parser.add_argument("--llrd_alpha", type=float, default=LR_DECAY,
                        help="LLRD layer-wise decay factor alpha")
    parser.add_argument("--focal_gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Starting threshold; dev-optimised during training")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(data_path: str, train_ids_path: str, dev_ids_path: str):
    """
    Load dontpatronizeme_pcl.tsv and split into train/dev DataFrames,
    preserving the original 0-4 severity scores for soft-label targets.
    """
    # Load raw TSV (skip the 4-line metadata header)
    raw = pd.read_csv(
        data_path, sep="\t", skiprows=4,
        names=["row_id", "par_id", "keyword", "country", "text", "orig_label"],
        dtype={"par_id": str, "orig_label": "Int64"},
        keep_default_na=False,
    )
    raw["orig_label"] = raw["orig_label"].astype(int)
    # Binary label: 0 if orig_label <= 1, else 1
    raw["label"] = (raw["orig_label"] >= 2).astype(int)

    def build_split(ids_path: str) -> pd.DataFrame:
        ids_df = pd.read_csv(ids_path, dtype={"par_id": str})
        ids_df.par_id = ids_df.par_id.astype(str)
        rows = []
        for pid in ids_df.par_id:
            row = raw.loc[raw.par_id == pid]
            if len(row) == 0:
                continue
            rows.append({
                "par_id":     pid,
                "keyword":    row.keyword.values[0],
                "text":       row.text.values[0],
                "label":      int(row.label.values[0]),
                "orig_label": int(row.orig_label.values[0]),
            })
        df = pd.DataFrame(rows)
        df["soft_target"] = df["orig_label"].map(SOFT_TARGET_MAP)
        return df

    trdf = build_split(train_ids_path)
    tedf = build_split(dev_ids_path)
    print(f"Train: {len(trdf)}  PCL={trdf.label.sum()}")
    print(f"Dev  : {len(tedf)}  PCL={tedf.label.sum()}")
    return trdf, tedf


# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class PCLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256,
                 use_soft_labels: bool = True):
        self.texts         = df["text"].tolist()
        self.labels        = (
            df["soft_target"].tolist() if use_soft_labels
            else df["label"].astype(float).tolist()
        )
        self.binary_labels = df["label"].tolist()
        self.tokenizer     = tokenizer
        self.max_len       = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "soft_label":     torch.tensor(self.labels[idx], dtype=torch.float),
            "binary_label":   torch.tensor(self.binary_labels[idx], dtype=torch.long),
        }


# ── Weighted sampler (oversamples PCL class) ───────────────────────────────────
def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Create a sampler that over-samples the minority (PCL) class.
    Motivated by EDA: only ~9.49% of examples are PCL across the full dataset.
    WeightedRandomSampler preserves all training examples (unlike undersampling).
    """
    class_counts = Counter(labels)
    n_total = len(labels)
    sample_weights = [n_total / class_counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=n_total, replacement=True
    )


# ── Focal Soft-BCE Loss ────────────────────────────────────────────────────────
class FocalSoftBCELoss(nn.Module):
    """
    Binary focal loss compatible with soft (non-integer) targets.

    Motivation: EDA shows label=1 examples are genuinely ambiguous. Soft targets
    give them a small positive gradient signal. Focal loss (Lin et al., 2017)
    down-weights easy examples and focuses capacity on hard/uncertain ones:

        L = -sum[ y*(1-p)^gamma*log(p) + (1-y)*p^gamma*log(1-p) ]

    where y is the *soft* target and gamma=2.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy(p, soft_targets, reduction="none")
        p_t = p * soft_targets + (1 - p) * (1 - soft_targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── Model ──────────────────────────────────────────────────────────────────────
class PCLClassifier(nn.Module):
    """DeBERTa-v3-base with a two-layer MLP classification head for PCL detection."""

    def __init__(self, model_name: str = MODEL_NAME, hidden_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        enc_hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(enc_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_rep = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_rep).squeeze(-1)   # (batch,)


# ── Grouped LLRD optimiser ─────────────────────────────────────────────────────
def get_llrd_optimizer(
    model: PCLClassifier,
    base_lr: float = 2e-5,
    head_lr: float = 1e-4,
    lr_decay: float = 0.9,
    weight_decay: float = 0.01,
):
    """
    Build AdamW parameter groups with layerwise LR decay (LLRD).

    - Embeddings: base_lr * lr_decay^(num_layers)
    - Layer i:    base_lr * lr_decay^(num_layers - i)
    - Head:       head_lr

    Lower LR for early layers → preserves general pre-trained representations.
    Higher LR for task head  → allows model to specialise.
    This technique is key to the PALI-NLP 1st-place SemEval-2022 system.
    """
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
    encoder  = model.encoder

    try:
        layers = encoder.encoder.layer
    except AttributeError:
        layers = encoder.encoder.layers

    num_layers = len(layers)
    param_groups = []

    # Embeddings — lowest LR
    embed_lr = base_lr * (lr_decay ** num_layers)
    for name, param in encoder.embeddings.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [param], "lr": embed_lr, "weight_decay": wd})

    # Each transformer layer — progressively higher LR
    for i, layer in enumerate(layers):
        layer_lr = base_lr * (lr_decay ** (num_layers - 1 - i))
        for name, param in layer.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            param_groups.append({"params": [param], "lr": layer_lr, "weight_decay": wd})

    # Classification head — highest LR
    for name, param in model.classifier.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [param], "lr": head_lr, "weight_decay": wd})

    optimizer = torch.optim.AdamW(param_groups)
    print(f"LLRD: embed_lr={embed_lr:.2e} | layer[0]={base_lr * lr_decay**(num_layers-1):.2e} "
          f"| layer[-1]={base_lr:.2e} | head={head_lr:.2e}")
    return optimizer


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, threshold: float = 0.5, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["binary_label"].numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= threshold).astype(int)
    f1_pcl   = f1_score(all_labels, preds, pos_label=1, average="binary", zero_division=0)
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    return {"f1_pcl": f1_pcl, "f1_macro": f1_macro, "logits": all_logits, "labels": all_labels}


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)

    # Load data
    trdf, tedf = load_data(args.data_path, args.train_ids, args.dev_ids)

    # Tokenizer and datasets
    tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = PCLDataset(trdf, tokenizer, max_len=args.max_length, use_soft_labels=True)
    dev_dataset   = PCLDataset(tedf, tokenizer, max_len=args.max_length, use_soft_labels=False)

    train_sampler = make_weighted_sampler(trdf["label"].tolist())
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               sampler=train_sampler, num_workers=2, pin_memory=True)
    dev_loader    = DataLoader(dev_dataset, batch_size=32, shuffle=False,
                               num_workers=2, pin_memory=True)

    print(f"Train batches: {len(train_loader)}  |  Dev batches: {len(dev_loader)}")

    # Model, loss, optimiser, scheduler
    model     = PCLClassifier(model_name=MODEL_NAME, dropout=DROPOUT).to(device)
    criterion = FocalSoftBCELoss(gamma=args.focal_gamma)
    optimizer = get_llrd_optimizer(
        model, base_lr=args.lr, head_lr=args.head_lr, lr_decay=args.llrd_alpha
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Total steps: {total_steps}  |  Warmup: {warmup_steps}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    best_f1   = 0.0
    best_ckpt = None
    history   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_labels    = batch["soft_label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, soft_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_steps    += 1

        avg_loss   = total_loss / n_steps
        dev_result = evaluate(model, dev_loader, threshold=0.5, device=device)
        history.append({"epoch": epoch, "loss": avg_loss,
                        "f1_pcl": dev_result["f1_pcl"],
                        "f1_macro": dev_result["f1_macro"]})

        print(f"Epoch {epoch}/{args.epochs}  "
              f"loss={avg_loss:.4f}  "
              f"dev_F1_PCL={dev_result['f1_pcl']:.4f}  "
              f"dev_F1_macro={dev_result['f1_macro']:.4f}")

        if dev_result["f1_pcl"] > best_f1:
            best_f1   = dev_result["f1_pcl"]
            best_ckpt = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest dev F1 (PCL, threshold=0.5): {best_f1:.4f}")

    # Threshold optimisation on dev set
    model.load_state_dict(best_ckpt)
    model.to(device)
    dev_eval   = evaluate(model, dev_loader, threshold=0.5, device=device)
    dev_logits = dev_eval["logits"]
    dev_labels = dev_eval["labels"]
    dev_probs  = torch.sigmoid(torch.tensor(dev_logits)).numpy()

    thresholds = np.arange(0.20, 0.81, 0.01)
    f1_scores  = [
        f1_score(dev_labels, (dev_probs >= t).astype(int),
                 pos_label=1, average="binary", zero_division=0)
        for t in thresholds
    ]
    best_thresh    = float(thresholds[np.argmax(f1_scores)])
    best_thresh_f1 = float(max(f1_scores))
    print(f"Optimal threshold: {best_thresh:.2f}  |  F1 PCL: {best_thresh_f1:.4f}")

    # Final classification report
    final_preds = (dev_probs >= best_thresh).astype(int)
    f1_macro_final = f1_score(dev_labels, final_preds, average="macro", zero_division=0)
    print(f"Dev Macro-F1 at optimal threshold: {f1_macro_final:.4f}")
    print(classification_report(dev_labels, final_preds, target_names=["No PCL", "PCL"]))

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "best_model_pcl.pt")
    torch.save(
        {
            "model_state_dict":  best_ckpt,
            "optimal_threshold": best_thresh,
            "dev_f1_pcl":        best_thresh_f1,
            "dev_f1_macro":      f1_macro_final,
            "history":           history,
            "hyperparameters": {
                "model_name":       MODEL_NAME,
                "max_len":          args.max_length,
                "epochs":           args.epochs,
                "base_lr":          args.lr,
                "head_lr":          args.head_lr,
                "lr_decay":         args.llrd_alpha,
                "focal_gamma":      args.focal_gamma,
                "soft_target_map":  SOFT_TARGET_MAP,
                "threshold":        best_thresh,
                "seed":             args.seed,
            },
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved → {ckpt_path}")
    print(f"Optimal threshold: {best_thresh:.2f}")
    print(f"Dev F1 (PCL class): {best_thresh_f1:.4f}")
    print(f"Dev Macro-F1:       {f1_macro_final:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
