import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prompts import get_prompt
from transformers import BitsAndBytesConfig


base_model_path = "Qwen/Qwen3-0.6B"
lora_checkpoint = "./qwen-domain-lora/checkpoint-232"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=bnb_config, trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_checkpoint)

model.eval()


"""base_model_path = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, device_map="auto")
model.eval()"""

user_inputs = [
    "before anything else, click the little settings gear in the top right",
    "type the whole sentence: automation is finally working",
    "start recording a new sequence for the login routine",
    "press ctrl shift escape to bring up task manager",
    "capture the screen once the popup appears",
    "double click the project folder named build_output",
    "wait 3.5 seconds before typing the password",
    "please focus on the main window again, it lost focus",
    "start looping this part until I tell you to stop",
    "write hello-world into the terminal and press enter",
    "click that bright blue confirm button next to the warning",
    "play the sequence that I saved earlier named cleanup",
    "ok now end the recording and store it",
    "sleep for a short moment, like barely a second",
    "press the spacebar twice to continue",
    "look at the screen and screenshot the whole desktop",
    "click on the document icon that looks like a gray sheet of paper",
    "type out the full email address including the domain",
    "stop looping immediately, it’s done processing",
    "press the escape key to cancel the selection",
    "double click open the installer in the downloads folder",
    "wait before continuing, maybe two or three seconds",
    "press enter to confirm the dialog",
    "start sequence recording for the cleanup phase now",
    "click on the green sync button below the calendar",
    "please capture the screen once the animation finishes",
    "write adminLogin into the username field",
    "stop loop and finalize this segment",
    "press f11 to toggle fullscreen",
    "type the command npm install and wait"
]


 
import time
for user_input in user_inputs:
    prompt = get_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.01,
            top_p=0.1
        )

    sequence_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    split_marker = "Convert this input into ONE valid automation command:"
    if split_marker in sequence_text:
        generated = sequence_text.split(split_marker, 1)[1].strip()
    else:
        generated = sequence_text.strip()  # fallback: use entire output
    print("Generated sequence:\n", generated)
    print("\n---------------------------------------------------------------------")
