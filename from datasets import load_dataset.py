from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

import torch

MODEL_NAME = "distilgpt2"
TRAIN_FILE = "gsm8k_train.csv"
TEST_FILE = "gsm8k_test.csv"
OUTPUT_DIR = "./gsm8k-distilgpt2"
EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 512

data_files = {"train": TRAIN_FILE, "validation": TEST_FILE}
dataset = load_dataset("csv", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['question'], truncation=True, padding=MAX_LENGTH)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["question", "answer"])  # remove unused columns
tokenized_datasets.set_format("torch")

training_args = TrainingArguments(
    output_dir="./gsm8k-distilgpt2",
    evaluation_strategy="epoch",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
)

trainer.train()