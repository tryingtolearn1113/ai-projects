import json
import torch
import os # Added to check local files
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

print("=== Starting AI Fine-Tuning ===")

# 1. Load your Flashcards
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

dataset = load_data("training_data.json")
print(f"Loaded {len(dataset)} examples (Flashcards)")

# 2. Load the Student (Offline!)
# We use your local folder so Windows doesn't crash
model_path = "./flan-t5-small"
print("Loading tokenizer and model offline...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

# 3. Add the "Sticky Notes" (LoRA)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # This will show you are only training ~1% of the brain!

# 4. Translate English to Computer Numbers
def tokenize(example):
    # Add clues: "Question: [text] Answer:"
    formatted_input = f"Question: {example['input']} Answer:"
    
    input_enc = tokenizer(
        formatted_input,
        max_length=128,
        padding="max_length",
        truncation=True
    )
    target_enc = tokenizer(
        example["output"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    # THE MAGIC FIX: Replace padding tokens with -100
    # This tells the AI to ignore the blank spaces!
    labels = []
    for token in target_enc["input_ids"]:
        if token == tokenizer.pad_token_id:
            labels.append(-100)
        else:
            labels.append(token)
            
    input_enc["labels"] = labels
    return input_enc

print("Translating words to tokens...")
tokenized_dataset = dataset.map(tokenize)

# 5. Set up the Teacher's Rules (UPDATED!)
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=50,             # INCREASED! Read the flashcards 50 times.
    learning_rate=3e-4,              # ADDED! Teacher speaks much louder for LoRA.
    per_device_train_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 6. Start studying!
print("Starting training... (Please wait a few minutes)")
trainer.train()
print("Training complete!")

# 7. Save the Sticky Notes
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
print("Model successfully saved to ./finetuned_model ✅")