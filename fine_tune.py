from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Use the valid model from Hugging Face
model_name = "cssupport/t5-small-awesome-text-to-sql"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load your dataset (use WikiSQL or custom dataset)
dataset = load_dataset("wikisql")


def preprocess(examples):
    inputs = [q for q in examples['question']]  # Natural language questions

    # Extract the human-readable SQL queries
    targets = [sql['human_readable'] for sql in examples['sql']]

    # Tokenize the natural language questions
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # Tokenize the SQL queries
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Apply the preprocessing function to the dataset
train_dataset = dataset.map(preprocess, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=train_dataset["validation"],  # Optional evaluation dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_text_to_sql_model")
tokenizer.save_pretrained("./fine_tuned_text_to_sql_model")
