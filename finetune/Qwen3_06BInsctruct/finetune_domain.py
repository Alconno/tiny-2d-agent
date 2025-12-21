import warnings, os, sys, datetime
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from prompts import get_prompt

checkpoint = None

BASE = "Qwen/Qwen3-0.6B"
DATA = r".\datasets\domain_06b\domain_06b_dataset.jsonl"
OUT_BASE = "./qwen-domain-lora"
seed = 412

# Generate a unique output folder for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(OUT_BASE, f"training_{timestamp}")
os.makedirs(OUT, exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True, use_cache=False)

# Dataset
dataset = load_dataset("json", data_files=DATA, split="train").shuffle(seed=seed)

def format_example(ex):
    prefix = get_prompt(ex["user_input"])
    full = prefix + ex["assistant_output"].strip() + "<|im_end|>"

    tok_full = tokenizer(full, truncation=True, max_length=768, padding="max_length")
    tok_prefix = tokenizer(prefix, truncation=True, max_length=768)

    input_ids = tok_full["input_ids"]
    attn = tok_full["attention_mask"]
    prefix_len = len(tok_prefix["input_ids"])

    labels = [-100] * len(input_ids)
    for i in range(prefix_len, len(input_ids)):
        if attn[i] == 1:
            labels[i] = input_ids[i]

    tok_full["labels"] = labels
    return tok_full

dataset = dataset.map(format_example, batched=False, remove_columns=dataset.column_names)

# LoRA setup
model = prepare_model_for_kbit_training(model)
if checkpoint and os.path.exists(checkpoint):
    model = PeftModel.from_pretrained(model, checkpoint)
else:
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora)

# Training
args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    logging_dir="./wandb",
    logging_strategy="steps",
    logging_steps=5,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    report_to="wandb"
)
trainer = Trainer(model=model, args=args, train_dataset=dataset)

if checkpoint and os.path.exists(checkpoint):
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# Save
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

print(f"Training completed. Files saved to: {OUT}")
