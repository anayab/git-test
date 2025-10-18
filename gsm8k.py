from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_NAME = "distilgpt2"
TRAIN_FILE = "gsm8k_train.csv"
TEST_FILE = "gsm8k_test.csv"
BATCH_SIZE = 1

dataset = load_dataset("csv", data_files={"train": TRAIN_FILE, "validation": TEST_FILE})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Concatenate question and answer as one training text
    text = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["question", "answer"])
tokenized_datasets.set_format("torch")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.config.pad_token_id = model.config.eos_token_id

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gsm8k-distilgpt2",
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()