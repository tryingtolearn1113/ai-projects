import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

print("=== Testing Fine-Tuned AI ===")

# 1. Paths to your folders
base_model_path = "./flan-t5-small"
lora_weights_path = "./finetuned_model"

# 2. Load the original "Base Brain" offline
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, use_fast=False)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path, local_files_only=True)

# 3. Add your custom "Sticky Notes" (LoRA weights)
print("Applying your EPICS fine-tuning weights...")
model = PeftModel.from_pretrained(base_model, lora_weights_path)

# Put the model in "Evaluation" mode (locks the brain so it just answers)
model.eval()
print("Model ready! ✅\n")

# 4. Create an asking function
def ask(question):
    # We add the exact same clues we used in training!
    formatted_question = f"Question: {question} Answer:"
    print(f"\n{formatted_question}")
    
    inputs = tokenizer(formatted_question, return_tensors="pt")
    
    # We add repetition_penalty just to keep it safe from looping
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        num_beams=2, 
        repetition_penalty=2.5
    )
    
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"AI Answer: {answer}")
    
    
# 5. Test it out!
ask("What does caget do in PyEPICS?")
ask("What is a PV in EPICS?")