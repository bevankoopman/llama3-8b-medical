from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
new_model = "./llama-3-8b-chat-doctor"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    new_model,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Patient"]},
               {"role": "assistant", "content": row["Doctor"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

question = ' '
while question!='':
    question=input("What is your question: ")

    messages = [
        {
            "role": "user",
            "content": question 
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                           add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                       truncation=True).to("cuda")
    
    outputs = model.generate(**inputs, max_length=150, 
                             num_return_sequences=1)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(text.split("assistant")[1])
    
