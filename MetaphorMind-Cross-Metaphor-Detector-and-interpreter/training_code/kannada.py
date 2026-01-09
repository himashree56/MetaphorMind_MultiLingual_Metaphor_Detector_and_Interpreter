# unified_xlm_roberta_kannada.py
# Kaggle-ready script: training + evaluation + misclassification analysis
# Usage (Kaggle): just run the cell (no args). Make sure dataset present at:
# /kaggle/input/kannada-data/kannada_dataset.json

import os
import sys
import json
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate

# ----------------------------
# Logging (file + stdout)
# ----------------------------
LOG_PATH = "/kaggle/working/training_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_log(*args, **kwargs):
    # convenience printing that also logs
    print(*args, **kwargs, flush=True)
    logger.info(" ".join(map(str, args)))

# Disable WandB if present
os.environ["WANDB_DISABLED"] = "true"

# ----------------------------
# Config (change if needed)
# ----------------------------
DATA_PATH = "/kaggle/input/kannada-data/kannada_dataset.json"
MODEL_NAME = "xlm-roberta-base"
OUTPUT_DIR = "/kaggle/working/xlm_kannada_metaphor"
SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 128
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
GRAD_ACCUM = 2  # effective batch size = PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACCUM * num_gpus
SAVE_REPORT = True
MISCLASSIFIED_CSV = "/kaggle/working/misclassified_analysis.csv"

# reproducibility
set_seed(SEED)

# ----------------------------
# Helper: load dataset robustly
# Accepts the format you described:
#   [ ["sent","label"], ["sent","label"], ... ]
# or possibly nested single element lists like [[["sent","label"], ...]]
# Also supports list with header (first row "sent","label")
# ----------------------------
def load_kannada_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle extra nesting like [[ ["sent","label"], ... ]]
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]

    # Expect raw to be list-of-lists
    if not isinstance(raw, list) or not all(isinstance(r, list) for r in raw):
        raise ValueError("Unexpected JSON structure. Expecting list of lists.")

    # If first row looks like header
    first0 = str(raw[0][0]).strip().lower() if raw and len(raw[0]) >= 2 else ""
    if first0 in ("sent", "sentence", "text"):
        rows = raw[1:]
    else:
        rows = raw

    # Normalize rows: some datasets might have single-element lists or 3 columns - we only take first two
    processed = []
    for r in rows:
        if not isinstance(r, list):
            continue
        if len(r) >= 2:
            sent = r[0]
            label = r[1]
            processed.append([str(sent).strip(), str(label).strip()])
    df = pd.DataFrame(processed, columns=["sentence", "label"])
    return df

# ----------------------------
# Load
# ----------------------------
print_log("PyTorch version:", torch.__version__)
print_log("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print_log("GPU name:", torch.cuda.get_device_name(0))
    except Exception:
        pass

print_log("Loading dataset from:", DATA_PATH)
df = load_kannada_json(DATA_PATH)
print_log("Total rows loaded:", len(df))
print_log("Sample rows:\n", df.head(5).to_dict(orient="records"))

# Basic cleaning
df['sentence'] = df['sentence'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip().str.lower()

# Validate minimal sanity
if df['sentence'].eq("").any():
    print_log("Warning: Some empty sentences found; these will be dropped.")
    df = df[~df['sentence'].eq("")].reset_index(drop=True)

# Map labels
label_map = {"normal": 0, "metaphor": 1}
if not set(df['label'].unique()).issubset(set(label_map.keys())):
    print_log("Warning: Found labels outside expected set:", df['label'].unique())
    # try to coerce common variants
    df['label'] = df['label'].replace({"metaphoric": "metaphor"})
    # If still invalid, raise
    if not set(df['label'].unique()).issubset(set(label_map.keys())):
        print_log("Error: Unexpected labels present. Expected 'normal' and 'metaphor'. Found:",
                  df['label'].unique())
        raise SystemExit(1)

df['label_id'] = df['label'].map(label_map)

# Shuffle and split
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['label_id'], random_state=SEED)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print_log(f"Train size: {len(train_df)}  Val size: {len(val_df)}")

# ----------------------------
# Tokenizer & Model
# ----------------------------
print_log("Loading tokenizer and model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print_log("Model loaded to device:", device)

# ----------------------------
# Build token frequency on train for heuristics
# ----------------------------
def build_token_freq(texts, tokenizer):
    cnt = Counter()
    for s in texts:
        toks = tokenizer.tokenize(str(s))
        cnt.update(toks)
    return cnt

token_freq = build_token_freq(train_df['sentence'].tolist(), tokenizer)

# ----------------------------
# Prepare datasets (tokenize)
# ----------------------------
def tokenize_batch(batch):
    return tokenizer(batch['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

train_ds = Dataset.from_pandas(train_df[['sentence','label_id']].rename(columns={'label_id':'label'}))
val_ds = Dataset.from_pandas(val_df[['sentence','label_id']].rename(columns={'label_id':'label'}))

print_log("Tokenizing datasets...")
train_ds = train_ds.map(lambda x: tokenizer(x['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
val_ds = val_ds.map(lambda x: tokenizer(x['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

# set format for Trainer
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
val_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

# ----------------------------
# Metrics
# ----------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# ----------------------------
# Training arguments (Kaggle-friendly)
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=False,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ----------------------------
# Train
# ----------------------------
print_log("Starting training...")
trainer.train()
print_log("Training finished. Saving best model & tokenizer to:", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ----------------------------
# Evaluate on validation and analyze misclassifications
# ----------------------------
print_log("Running predictions on validation set for detailed analysis...")
pred_output = trainer.predict(val_ds)
logits = pred_output.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
preds = np.argmax(logits, axis=1)
confidences = probs.max(axis=1)

# Prepare error analysis
rows = []
train_sent_set = set(train_df['sentence'].astype(str).tolist())
train_label_map = defaultdict(set)
for s, l in zip(train_df['sentence'].astype(str), train_df['label']):
    train_label_map[s].add(l)
majority_label = train_df['label'].value_counts().idxmax()

for i, r in val_df.reset_index(drop=True).iterrows():
    sent = str(r['sentence'])
    true_label = r['label']
    true_id = r['label_id']
    pred_id = int(preds[i])
    pred_label = [k for k,v in label_map.items() if v==pred_id][0]
    prob = float(confidences[i])
    tokens = tokenizer.tokenize(sent)
    num_tokens = len(tokens)
    len_chars = len(sent)
    rare_frac = 0.0
    if num_tokens > 0:
        rare_count = sum(1 for t in tokens if token_freq.get(t, 0) <= 1)
        rare_frac = rare_count / num_tokens
    in_train = sent in train_sent_set
    conflict_in_train = len(train_label_map.get(sent, set())) > 1

    misclassified = (pred_label != true_label)
    rows.append({
        "sentence": sent,
        "true_label": true_label,
        "pred_label": pred_label,
        "prob": prob,
        "num_tokens": num_tokens,
        "len_chars": len_chars,
        "rare_token_frac": rare_frac,
        "in_train": in_train,
        "conflict_in_train": conflict_in_train,
        "misclassified": misclassified
    })

analysis_df = pd.DataFrame(rows)
mis_df = analysis_df[analysis_df['misclassified']].copy()

# Add heuristic categories for misclassifications
def categorize_row(row):
    cats = []
    if row['prob'] < 0.6:
        cats.append('low_confidence')
    if row['rare_token_frac'] > 0.5:
        cats.append('rare_tokens')
    if row['num_tokens'] < 3:
        cats.append('too_short')
    if row['num_tokens'] > 60:
        cats.append('too_long')
    if not row['in_train']:
        cats.append('no_similar_in_train')
    if row['conflict_in_train']:
        cats.append('label_conflict_in_train')
    if row['pred_label'] == majority_label:
        cats.append('predicted_majority_class')
    if not cats:
        cats.append('other')
    return ";".join(cats)

if not mis_df.empty:
    mis_df['categories'] = mis_df.apply(categorize_row, axis=1)
    if SAVE_REPORT:
        mis_df.to_csv(MISCLASSIFIED_CSV, index=False)
    # summary counts
    cat_counts = Counter()
    for cats in mis_df['categories'].fillna(''):
        for c in str(cats).split(';'):
            if c:
                cat_counts[c] += 1

    print_log(f"Total validation samples: {len(val_df)}")
    print_log(f"Misclassified on validation: {len(mis_df)}")
    print_log("Error category counts:")
    for k,v in cat_counts.most_common():
        print_log(f" - {k}: {v}")

    # show examples (up to 5 each)
    print_log("\nExamples by category (up to 5 each):")
    for cat in cat_counts.keys():
        print_log(f"\n== {cat} ==")
        subset = mis_df[mis_df['categories'].str.contains(cat)].head(5)
        for _, r in subset.iterrows():
            print_log(f"True: {r['true_label']}  Pred: {r['pred_label']}  p={r['prob']:.2f} tokens={r['num_tokens']} rare_frac={r['rare_token_frac']:.2f}")
            print_log(" ", r['sentence'])
else:
    print_log("No misclassifications found on validation set!")

# ----------------------------
# Overall classification report
# ----------------------------
print_log("\nFull classification report on validation set:")
print_log(classification_report(val_df['label_id'], preds, target_names=['normal','metaphor']))

# ----------------------------
# Save summary outputs
# ----------------------------
print_log("Saving final artifacts...")
# misclassified CSV saved above
MODEL_SAVE_DIR = "/kaggle/working/saved_model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
trainer.save_model(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print_log("Saved model/tokenizer to", MODEL_SAVE_DIR)
print_log("Saved misclassification report to", MISCLASSIFIED_CSV)
print_log("Training log:", LOG_PATH)

print_log("All done.")
