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
from sklearn.model_selection import train_test_split
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
print_flush("2 - IndicBERT (ai4bharat/indic-bert)")
while True:
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice in ["1", "2"]:
        break
    print_flush("Invalid choice, please enter 1 or 2.")

if choice == "1":
    model_name = "xlm-roberta-base"
elif choice == "2":
    model_name = "ai4bharat/indic-bert"
else:
    print_flush("Invalid choice, defaulting to IndicBERT.")
    model_name = "ai4bharat/indic-bert"

print_flush(f"\n✅ Using model: {model_name}")

# =====================================================
# 3. Load dataset
# =====================================================
data_path = "/kaggle/input/tamil-data/TAMIL_DATASET.json"
print_flush(f"Loading dataset from {data_path}...")
try:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print_flush("Dataset loaded successfully.")
    print_flush(f"Data sample: {data[:2]}")
except FileNotFoundError as e:
    print_flush(f"Error: Dataset file not found - {e}")
    sys.exit(1)

# Convert to DataFrame
print_flush("Converting to DataFrame...")
try:
    df = pd.DataFrame(data, columns=["sentence", "label"])
    print_flush(f"DataFrame shape: {df.shape}")
    print_flush(f"DataFrame sample:\n{df.head()}")
except ValueError as e:
    print_flush(f"Error: Failed to create DataFrame - {e}")
    sys.exit(1)

# Split into train and test (80/20)
print_flush("Splitting dataset into train and test...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print_flush(f"Train DataFrame shape: {train_df.shape}")
print_flush(f"Test DataFrame shape: {test_df.shape}")

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
print_flush("விழி பயிற்சி செய்கிறார்। ->", predict_sentence("விழி பயிற்சி செய்கிறார்।"))
print_flush("அவள் கண்கள் கடல் ஆழம் கொண்டவை। ->", predict_sentence("அவள் கண்கள் கடல் ஆழம் கொண்டவை।"))

# =====================================================
# 11. Interactive Prediction Loop
# =====================================================
print_flush("\n🔮 Interactive Sentence Prediction")
print_flush("Enter a Tamil sentence to classify (or type 'exit' to quit):")
while True:
    try:
        sentence = input("> ").strip()
        if sentence.lower() == "exit":
            print_flush("Exiting prediction loop.")
            break
        prediction = predict_sentence(sentence)
        print_flush(f"Sentence: {sentence}")
        print_flush(f"Prediction: {prediction}\n")
    except KeyboardInterrupt:
        print_flush("Prediction loop interrupted. Exiting.")
        break
    except Exception as e:
        print_flush(f"Error during prediction: {e}")
        continue

# Save model for later use
print_flush("Saving model...")
model.save_pretrained("/kaggle/working/saved_model")
tokenizer.save_pretrained("/kaggle/working/saved_model")
print_flush("Model and tokenizer saved to /kaggle/working/saved_model")