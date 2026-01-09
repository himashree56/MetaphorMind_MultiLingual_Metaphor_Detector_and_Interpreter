import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
from sklearn.metrics import classification_report
import sys
import torch
import os
import logging

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("/kaggle/working/training_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force print output
def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    logger.info(" ".join(map(str, args)))

# Explicitly disable WandB
os.environ["WANDB_DISABLED"] = "true"

# =====================================================
# 1. Check Resources
# =====================================================
print_flush(f"PyTorch version: {torch.__version__}")
print_flush(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print_flush(f"GPU name: {torch.cuda.get_device_name(0)}")
    print_flush(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print_flush(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# =====================================================
# 2. USER INPUT: Choose model
# =====================================================
print_flush("Choose a model to train:")
print_flush("1 - XLM-RoBERTa (xlm-roberta-base)")
print_flush("2 - IndicBERT v2 (ai4bharat/indic-bert-v2)")

# Hardcode for testing
choice = "1"
if choice == "1":
    model_name = "xlm-roberta-base"
elif choice == "2":
    model_name = "ai4bharat/indic-bert-v2"
else:
    print_flush("Invalid choice, defaulting to XLM-RoBERTa.")
    model_name = "xlm-roberta-base"

print_flush(f"\n✅ Using model: {model_name}")

# =====================================================
# 3. Load dataset
# =====================================================
print_flush("Loading datasets...")
try:
    with open("/kaggle/input/hindi-data/hindi_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("/kaggle/input/hindi-data/hindi_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print_flush("Datasets loaded successfully.")
    print_flush(f"Train data sample: {train_data[:2]}")
    print_flush(f"Test data sample: {test_data[:2]}")
except FileNotFoundError as e:
    print_flush(f"Error: Dataset file not found - {e}")
    sys.exit(1)

# Convert to DataFrames
print_flush("Converting to DataFrames...")
try:
    train_df = pd.DataFrame(train_data, columns=["sentence", "label"])
    test_df = pd.DataFrame(test_data, columns=["sentence", "label"])
    print_flush(f"Train DataFrame shape: {train_df.shape}")
    print_flush(f"Test DataFrame shape: {test_df.shape}")
    print_flush(f"Train DataFrame sample:\n{train_df.head()}")
except ValueError as e:
    print_flush(f"Error: Failed to create DataFrame - {e}")
    sys.exit(1)

# Validate dataset
print_flush("Validating dataset...")
if train_df["sentence"].str.strip().eq("").any() or test_df["sentence"].str.strip().eq("").any():
    print_flush("Error: Empty sentences found in dataset.")
    sys.exit(1)

# Map labels to integers
label_map = {"normal": 0, "metaphor": 1}
train_df["label"] = train_df["label"].map(label_map)
test_df["label"] = test_df["label"].map(label_map)

if train_df["label"].isnull().any() or test_df["label"].isnull().any():
    print_flush("Error: Invalid labels found in dataset.")
    sys.exit(1)

# Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
print_flush(f"Train dataset size: {len(train_dataset)}")
print_flush(f"Test dataset size: {len(test_dataset)}")

# =====================================================
# 4. Tokenizer + Model
# =====================================================
print_flush("Loading tokenizer and model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    print_flush("Tokenizer and model loaded successfully.")
    print_flush(f"Model device: {model.device}")
except Exception as e:
    print_flush(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)

print_flush("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print_flush("Datasets tokenized and formatted.")

# =====================================================
# 5. Metrics
# =====================================================
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    metrics = {}
    metrics["accuracy"] = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    metrics["f1"] = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return metrics

# =====================================================
# 6. Training Arguments
# =====================================================
training_args = TrainingArguments(
    output_dir="/kaggle/working/results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,  # Frequent logging
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Further reduced batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 4*4 = 16
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",  # Explicitly disable WandB
    push_to_hub=False,
    fp16=torch.cuda.is_available(),  # Enable mixed precision for GPU
)

# =====================================================
# 7. Trainer
# =====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,  # Fix FutureWarning
    compute_metrics=compute_metrics,
)

# =====================================================
# 8. Train & Evaluate
# =====================================================
try:
    print_flush("Starting training...")
    trainer.train()
    print_flush("Training completed!")
except Exception as e:
    print_flush(f"Training failed with error: {e}")
    raise

print_flush("\nEvaluating model...")
eval_results = trainer.evaluate()
print_flush("\n📊 Evaluation results:", eval_results)

# =====================================================
# 9. Classification Report
# =====================================================
print_flush("Generating classification report...")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

print_flush("\nClassification Report:")
print(classification_report(test_df["label"], preds, target_names=["normal", "metaphor"]))

# =====================================================
# 10. Example Predictions
# =====================================================
def predict_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    return "normal" if pred == 0 else "metaphor"

print_flush("\n🔮 Prediction Examples:")
print_flush("दिल पत्थर का है। ->", predict_sentence("दिल पत्थर का है।"))
print_flush("राम बाजार गया। ->", predict_sentence("राम बाजार गया।"))

# Save model for later use
print_flush("Saving model...")
model.save_pretrained("/kaggle/working/saved_model")
tokenizer.save_pretrained("/kaggle/working/saved_model")
print_flush("Model and tokenizer saved to /kaggle/working/saved_model")