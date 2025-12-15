# ---- Standard / Third-party ----
import os, re, time, threading
from collections import deque
from peft import PeftModel
from access_models import AccessModels

# ---- Local / project modules ----
from VoiceTranscriber import VoiceTranscriber
from utility.dogshitretard import (
    embd_ocr_lines, extract_action, embd_events,
    extract_target_context, parse_delay, image_diff_percent,
    image_hash, filter_numbers
)
from utility.image_matching import find_crop_in_image
from events.possible_events import possible_events
from events.Mouse import Mouse, MouseButton
from events.Keyboard import Keyboard, KeyboardEvent
from events.SequenceProcessor import SequenceEvent, SequenceProcessor
from events.Timer import Timer
from events.ScreenCapture import ScreenCapture, ScreenCaptureEvent
from events.Conditioning import Condition, ConditionProcessor
from events.Variable import VariableEvent, VariableProcessor
from events.Loop import LoopProcessor, LoopEvent
from class_models.Context import Context
from class_models.Variable import Variable

# ---- Logging / environment ----
import logging
logging.getLogger("ppocr").setLevel(logging.WARNING)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---- Models & events ----
models = AccessModels()
sequenceProcessor = SequenceProcessor(models.embd_func, "sequences.json")
keyboard = Keyboard(models.embd_func)
mouse = Mouse()
conditionProcessor = ConditionProcessor()
variableProcessor = VariableProcessor()
loopProcessor = LoopProcessor() 
#print("pre embedding possible events", possible_events)
event_embeds = embd_events(models.embd_func, possible_events)
context_queue = deque()


# ---- Voice listener ----
voiceTranscriber = VoiceTranscriber()
pause_listener_event = threading.Event()
pause_listener_event.set()

def context_listener():
    while True:
        if pause_listener_event.is_set():
            voice_input = voiceTranscriber()
            if voice_input and len(voice_input)>0:
                print("New voice input: ", voice_input)
                context_queue.append(Context(voice_input))
        else:
            time.sleep(0.1)

threading.Thread(target=context_listener, daemon=True).start()



# ---- State ----
n_retries = 4
current_context = Context("", n_retries)
recording_state = {"active": False, "contexts": [], "name": ""}
recording_stack = [recording_state["contexts"]]
use_GPT = False 

screen_capture = ScreenCapture()
screenshot_box = None
prev_hash = None
prev_preds = None
prev_embd_lines = None
prev_screenshot = None
screenshot_sim_threshold = 0.001

variables: dict = {}

colors = ["black","white","red","green","blue","yellow","orange","brown","gray","purple"]
color_pattern = r"\b(" + "|".join(colors) + r")\b"

# ---- Helper functions ----
def append_to_recording_seq(curr_ctx: Context, is_template, found_colors, action_result):
    section = recording_stack[-1]
    
    if is_template:
        color = "" if found_colors is None else found_colors.group(1).strip()
        var_name = is_template.group(1).replace(color, "").strip()
        curr_ctx.text = f"{action_result.get('span','Type').strip()} {color} {{{{{var_name}}}}}"

    if curr_ctx.text: curr_ctx.text = curr_ctx.text.strip()
    section.append(curr_ctx)
    
def append_condition_to_recording_seq(curr_ctx: Context):
    global recording_stack, recording_state
    if recording_state["active"]:
        curr_ctx.meta = {"if": condition}
        curr_ctx.sub_contexts = []

        parent = recording_stack[-1]
        if curr_ctx in parent:
            raise RuntimeError("Cycle detected: curr_ctx already exists in its parent")
        if parent is curr_ctx.sub_contexts:
            raise RuntimeError("Cycle detected: parent list equals child's sub_contexts")
        
        parent.append(curr_ctx)
        recording_stack.append(curr_ctx.sub_contexts)

def run_ocr(screenshot, offset, found_colors, number_only=False):
    global prev_hash, prev_preds, prev_embd_lines, prev_screenshot

    diff = image_diff_percent(screenshot, prev_screenshot) if prev_hash else 1e8
    if diff < screenshot_sim_threshold:
        preds, embd_lines = prev_preds, prev_embd_lines
    else:
        preds = models.ocr_func(screenshot, offset)  
        if found_colors:
            preds = [[
                    (bbox, text, color) for bbox, text, color in line
                    if color in found_colors 
                ]for line in preds]

        print("preds: ", preds)
        if number_only: preds = filter_numbers(preds)
        embd_lines = embd_ocr_lines(models.embd_func, preds, found_colors)
        prev_screenshot, prev_preds, prev_embd_lines, prev_hash = screenshot, preds, embd_lines, image_hash(screenshot)
    return embd_lines


def apply_gpt_to_context(ctx):
    if models.gpt_func:
        res = models.gpt_func(ctx).split("\n")
        ctx_processed = res[0].strip() if res else ""
        for line in res[1:]:
            context_queue.append(line.strip())
        return ctx_processed
    return ctx

# ---- Main loop ----
print("\n\nMain loop started, hold F8 to say your commands!\n\n")
parsed_action_cache = {}      # orig_ctx -> parsed action
retry_target_cache = {}       # orig_ctx -> (target_text, found_colors)
retries = {}    
while True:
    # --- Fetch new context ---
    if current_context and (current_context.text == "" or current_context.text == None) and context_queue:
        current_context = context_queue.popleft()  

    if isinstance(current_context, str):
        current_context = Context()

    if current_context == None or current_context.text == "" or current_context.text == None:
        time.sleep(0.1)
        continue

    orig_ctx = current_context.text

    # ---- GPT & action parsing (once per context) ----
    if orig_ctx not in parsed_action_cache:
        ctx_processed = apply_gpt_to_context(orig_ctx) if use_GPT else orig_ctx
        ctx_processed = re.sub(r"[.,;!?]", "", ctx_processed).strip()

        found_colors = re.findall(color_pattern, ctx_processed, re.I)
        found_colors = [c.lower() for c in found_colors] or None

        parsed_action = extract_action(ctx_processed, event_embeds, models.embd_func)
        if not parsed_action:
            print("Could not extract action")
            current_context = ""
            continue

        action_result = parsed_action["result"]
        target_text = extract_target_context(parsed_action.get("span",""), ctx_processed)

        parsed_action_cache[orig_ctx] = parsed_action
        retry_target_cache[orig_ctx] = (target_text, found_colors)
    else:
        parsed_action = parsed_action_cache[orig_ctx]
        action_result = parsed_action["result"]
        target_text, found_colors = retry_target_cache[orig_ctx]

    print(">>>>>>>>>>>>> context:", orig_ctx)
    print(">>>>>>>>>>>>> action:", action_result)

    # ---- Template check ----
    is_template = re.match(
        r"(.+?)\s+(?:as var|variable|as variable|as a template|as a variable|template|is variable|as template|is template)$", 
        target_text, re.I
    )
    if is_template and not recording_state["active"]:
        print("You cannot use templates while not recording")
        current_context = Context()
        continue

    failed = True
    # ---- Event processing ----
    if not is_template:
        if isinstance(action_result, MouseButton):
            if target_text:
                failed = mouse.process_event(parsed_action, target_text, variables, screenshot_box, found_colors, 
                    embd_func=models.embd_func, run_ocr_func=run_ocr)

        elif isinstance(action_result, KeyboardEvent):
            if target_text:
                keyboard.execute(action_result, target_text)
                failed = False

        elif isinstance(action_result, Timer):
            delay_s = parse_delay(target_text)
            if delay_s:
                time.sleep(delay_s)
                failed = False

        elif isinstance(action_result, ScreenCaptureEvent):
            print(action_result)
            print(target_text)
            if target_text:
                nums = list(map(int, re.findall(r"\d+", target_text)))
                if len(nums) != 4:
                    print("Invalid screenshot box format")
                    failed = True
                screenshot_box = tuple(nums)
            else:
                _, screenshot_box = screen_capture.select_region()
                time.sleep(0.25)
                current_context.text += " {},{},{},{}".format(*screenshot_box)
            print("current box: ", screenshot_box)
            failed = False
                    
    # Variables
    if isinstance(action_result, VariableEvent):
        if target_text:
            var = variableProcessor.process_event(parsed_action, target_text, screenshot_box, found_colors, 
                    embd_func=models.embd_func, run_ocr_func=run_ocr)
            variables[var.name] = var
            print(var)
            failed = False

    # Looping
    if isinstance(action_result, LoopEvent):
        failed, recording_stack = loopProcessor.process(action_result, target_text, current_context,
                              recording_stack, is_template)

    # Sequence recording
    if recording_state["active"] and action_result \
                and not isinstance(action_result, SequenceEvent) \
                and not isinstance(action_result, Condition) \
                and not isinstance(action_result, LoopEvent):
        print("parsed action: ", parsed_action)
        append_to_recording_seq(current_context, is_template, found_colors, parsed_action)
        failed = False

    # Sequence event handling 
    if isinstance(action_result, SequenceEvent):
        if target_text or action_result in (SequenceEvent.SAVE):
            failed, recording_state, recording_stack = \
                sequenceProcessor.process_sequence_event(
                    action_result, target_text, recording_state, recording_stack,
                    voiceTranscriber, pause_listener_event, context_queue,
                )
    # Condition event handling
    elif isinstance(action_result, Condition):
        if action_result == Condition.IF:
            condition = conditionProcessor.parse_condition(target_text) # {type, query, colors, negate}
            if condition:
                if condition["type"] == "text": 
                    passed, target = conditionProcessor.check_text(condition, screenshot_box, run_ocr_func=run_ocr, embd_func=models.embd_func)
                    append_condition_to_recording_seq(current_context)
                elif condition["type"] == "image":
                    passed, target = conditionProcessor.check_image(condition, screenshot_box, embd_func=models.embd_func)
                    append_condition_to_recording_seq(current_context)
                elif condition["type"] == "variable":
                    passed, var_obj = conditionProcessor.check_variable(condition, variables, embd_func=models.embd_func)
                    append_condition_to_recording_seq(current_context)
                if passed:
                    failed = False
                    if current_context.sub_contexts and not recording_state["active"]:
                        context_queue.extend(current_context.sub_contexts)
        elif action_result == Condition.END_IF:
            if len(recording_stack) > 1: recording_stack.pop()
            failed = False


    # Retry logic 
    if failed:
        retries[orig_ctx] = retries.get(orig_ctx, n_retries) - 1
        if retries[orig_ctx] > 0:
            print(f"Retrying {orig_ctx} ({retries[orig_ctx]} left)")
            context_queue.appendleft(current_context)
        else:
            print(f"Skipping {orig_ctx}")
            retries.pop(orig_ctx, None)
            parsed_action_cache.pop(orig_ctx, None)
            retry_target_cache.pop(orig_ctx, None)

        current_context = Context()
        time.sleep(1)
        continue

    # ---- Clear caches on success ----
    current_context = Context()
    time.sleep(0.2)
