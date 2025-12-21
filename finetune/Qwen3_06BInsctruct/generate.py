import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch


def setup_gpt_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from prompts import get_prompt

    global _model, _get_prompt, _tokenizer

    _get_prompt = get_prompt
    _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    _model = PeftModel.from_pretrained(base_model, "./qwen-domain-lora/training_20251218_032425/checkpoint-292")
    _model.eval()

    print(f"---- Hosting GPT model: {sum(p.numel() for p in _model.parameters())} params ----")

    return _model, _tokenizer, _get_prompt

def generate(user_input: str, _model, _tokenizer, _get_prompt) -> str:
    if not _model or not _tokenizer or not _get_prompt:
        setup_gpt_model()
    time.sleep(0.01)

    prompt = _get_prompt(user_input)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    print("GPT CALL..")
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.01, top_p=0.1)
    text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("assistant\n", 1)[1].strip() \
        if "assistant\n" in text else text.strip()
