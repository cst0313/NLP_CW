# predict.py
# ─────────────────────────────────────────────────────────────────────────────
# Generates predictions from a saved DeBERTa-v3-base checkpoint
# for SemEval-2022 Task 4 Subtask 1 (Patronizing and Condescending Language).
#
# Usage:
#   # Dev set predictions
#   python predict.py \
#     --data_path dontpatronizeme_pcl.tsv \
#     --split_ids data/splits/dev_semeval_parids-labels.csv \
#     --model_path BestModel/checkpoint/best_model_pcl.pt \
#     --output dev.txt \
#     --threshold 0.35
#
#   # Test set predictions (raw TSV, no labels)
#   python predict.py \
#     --data_path data/raw/task4_test.tsv \
#     --model_path BestModel/checkpoint/best_model_pcl.pt \
#     --output test.txt \
#     --threshold 0.35
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.WARNING)

# ── Defaults matching train.py ─────────────────────────────────────────────────
MAX_LENGTH = 256
DROPOUT    = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PCL predictions from a saved DeBERTa checkpoint"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to dontpatronizeme_pcl.tsv (for labelled splits) "
             "or task4_test.tsv (for the unlabelled test set)"
    )
    parser.add_argument(
        "--split_ids", type=str, default=None,
        help="CSV with par_id column to select examples from --data_path. "
             "Omit when --data_path is already task4_test.tsv (no header, no split needed)."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to saved model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output file, e.g. dev.txt or test.txt. One prediction per line."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Classification threshold (default: 0.35, dev-optimised)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    return parser.parse_args()


# ── Model ──────────────────────────────────────────────────────────────────────
class PCLClassifier(nn.Module):
    """DeBERTa-v3-base with a two-layer MLP classification head."""

    def __init__(self, model_name: str, hidden_size: int = 256,
                 dropout: float = DROPOUT):
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
        return self.classifier(out.last_hidden_state[:, 0, :]).squeeze(-1)


# ── Inference dataset ──────────────────────────────────────────────────────────
class PCLInferenceDataset(Dataset):
    def __init__(self, texts: list, tokenizer, max_len: int = 256):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
        }


# ── Data loading ───────────────────────────────────────────────────────────────
def load_texts(data_path: str, split_ids_path: str = None) -> list:
    """
    Load texts from data_path.

    If split_ids_path is provided, reads par_ids from it and looks them up
    in data_path (assumed to be dontpatronizeme_pcl.tsv format).

    If split_ids_path is None, assumes data_path is the test TSV
    (columns: idx, par_id, keyword, country, text — no header).
    """
    if split_ids_path is not None:
        # Labelled split: look up by par_id in the main TSV
        raw = pd.read_csv(
            data_path, sep="\t", skiprows=4,
            names=["row_id", "par_id", "keyword", "country", "text", "orig_label"],
            dtype={"par_id": str},
            keep_default_na=False,
        )
        ids_df = pd.read_csv(split_ids_path, dtype={"par_id": str})
        ids_df.par_id = ids_df.par_id.astype(str)
        texts = []
        for pid in ids_df.par_id:
            row = raw.loc[raw.par_id == pid]
            if len(row) == 0:
                print(f"WARNING: par_id {pid} not found in TSV — using empty string")
                texts.append("")
            else:
                texts.append(row.text.values[0])
    else:
        # Unlabelled test TSV (no header, 5 columns)
        df = pd.read_csv(
            data_path, sep="\t", header=None,
            names=["idx", "par_id", "keyword", "country", "text"],
            keep_default_na=False,
        )
        texts = df["text"].tolist()

    print(f"Loaded {len(texts)} texts from {data_path}")
    return texts


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_binary(model, loader, threshold: float, device) -> list:
    """Return a flat list of binary predictions (0/1) for every example."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= threshold).astype(int).tolist()
            all_preds.extend(preds)
    return all_preds


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.model_path}")
    ckpt       = torch.load(args.model_path, map_location="cpu", weights_only=False)
    hp         = ckpt.get("hyperparameters", {})
    model_name = hp.get("model_name", "microsoft/deberta-v3-base")
    saved_thr  = ckpt.get("optimal_threshold", args.threshold)

    # Use threshold from CLI if explicitly set, else fall back to saved value
    threshold = args.threshold if args.threshold != 0.35 else saved_thr
    print(f"Using threshold: {threshold:.2f}  (checkpoint saved: {saved_thr:.2f})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = PCLClassifier(model_name=model_name, dropout=DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Model loaded.")

    # Load texts
    texts = load_texts(args.data_path, args.split_ids)

    # Build DataLoader
    dataset = PCLInferenceDataset(texts, tokenizer, max_len=args.max_length)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    # Run inference
    preds = predict_binary(model, loader, threshold=threshold, device=device)

    # Write output
    with open(args.output, "w") as f:
        f.write("\n".join(str(p) for p in preds) + "\n")

    pcl_count  = sum(preds)
    npcl_count = len(preds) - pcl_count
    print(f"Predictions saved → {args.output}")
    print(f"  {len(preds)} lines  |  PCL={pcl_count}  No-PCL={npcl_count}")


if __name__ == "__main__":
    main()
