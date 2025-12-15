from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, warnings, wandb

warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False),
    device_map="auto",
    trust_remote_code=True,
)

dataset = load_dataset("tatsu-lab/alpaca")

def format_prompt(ex):
    s="You are a helpful assistant."
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{ex['instruction']} - {ex['input']}<|im_end|>\n<|im_start|>assistant\n{ex['output']}<|im_end|>"

dataset = dataset.map(lambda x: {"prompt": format_prompt(x)})

def tokenize_fn(batch):
    r = tokenizer(batch["prompt"], truncation=True, max_length=600, padding="max_length")
    r["labels"] = r["input_ids"].copy()
    return r

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

wandb.init(project="qwen3-instruct")
training_args = TrainingArguments(
    output_dir="./qwen-alpaca-lora", per_device_train_batch_size=8, gradient_accumulation_steps=2,
    gradient_checkpointing=False, logging_dir="./wandb",
    logging_strategy="steps", logging_steps=1, num_train_epochs=3, learning_rate=2e-4,
    fp16=True, save_strategy="epoch", report_to="wandb"
)
tokenized_dataset["train"] = tokenized_dataset["train"].select(range(1500))

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"])
trainer.train()
