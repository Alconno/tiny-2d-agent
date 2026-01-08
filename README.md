---

A **local, deterministic desktop RPA agent** that turns natural language  
(**voice or text**) into **real UI actions** using OCR, image understanding, and a constrained action space.


---

## 🎥 Demo (recommended)

▶️ **YouTube demo**  
Shows voice commands, OCR, spatial reasoning, retries, and real desktop interaction.

👉 https://youtu.be/mXx7YiOWfbA

---

## 1️⃣ Install requirements
```bash
pip install -r requirements.txt
```

---

## 2️⃣ Finetune the model (local only)
Before running the agent, you need to fine-tune the GPT model locally:
```bash
python .\finetune\Qwen3_06BInsctruct\finetune_domain.py
```
> No cloud upload is required. Do not upload to Hugging Face or any external service.

---

## 3️⃣ Host models
Start the local model server:
```bash
python .\fastAPI\host_models.py
```

---

## 4️⃣ Run the voice-driven agent
```bash
python ./main.py
```
- Wait until it logs that it is listening.  
- Hold **F8** to say commands **one at a time**.  

---

## 5️⃣ Commands

### Mouse / UI actions
> Targets can also have "color" or "color or color..." behind them -> example: "click blue or red target"
- `click target`  
- `right click target`  
- `double click target`  
- `shift click target`  
- `shift right click target`  
- `click image` / `right click image` / `double click image`  
- `click coord x y` — click at coordinates  
- `click all variable variable_name` — clicks all values inside a variable  
- `click top variable variable_name` — clicks the top value  
- Spatial awarness: 
  - `click left of target` / `click right of target` / `click above of target` / `click below of target`  
> Can also be used with text or empty areas  

#### Template usage:
- Template is a variable used only when playing the recording:
  - When playing the recording, it will ask you to enter the value for "template_name" unless you predefined it in the json
  - "Click target as template"
  - "Start loop items as template" (GPT automatically generates this command for simply saying "loop over items")
  - "Write text as template"
- Inside the json, they will be saved like {{target}}, {{items}}... in the sequence
- They will have their data below the sequence recording inside "vars" list
- Templates can be string, numeric, lists, lists of lists... and each list can contain different values: ["yes", 4, [5,5]]
- List templates are accessed with {{target.0.0}} for a nested list in list

### Keyboard / Text
- `write <text>` / `type <text>`  
- `press <key>`  

### Recording sequences
- `start recording name_of_recording` — starts recording actions into `sequences.json`  
- `stop recording` — stops the last recording  
- `play recording name_of_recording` — plays a saved sequence  
- `reset recording` — clears all recorded steps  
- `clear previous step` — removes last step  

### Variables
- `set variable <name>` — sets a variable for later use in conditions or clicks  
> Variables support numeric ranges (`>10<30`) and coloring (`all values red below 10`)  

### Loops
- `start loop <number>` — numeric loop  
- `start loop over <items>` — loop over template variable items  
- `end loop`  

### Conditional execution
- `if <condition>`  
- `end if`  

### Timing / Waiting
- `wait <seconds>` — sleep  
- `waitFor <text_or_image> <timeout>` — wait for text or image to appear  

### GPT control
- `toggle GPT on/off`  

> Some commands are self-explanatory

---

## 6️⃣ Using the API (programmatic access)
If you know even a little programming, you can bypass voice entirely and access all runtime data, enabling **insane automation possibilities**.

```python
from API import API

api = API()

# Focus on an area
focus_area = api("focus")

# Click on processing target
retdata = api("click empty box below of processing")
somevar = api("set variable named burgers with values yellow below 50")

# Do anything with data and new vars, loops, complex conditions, sorts etc...
```

- API returns full runtime data, including context, event info, and processed targets
- You can combine sequences, variables, and logic however you want
- Powerful for custom automations that voice alone cannot easily do

