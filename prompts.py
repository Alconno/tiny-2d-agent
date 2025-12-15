from events.possible_events import possible_events

event_mapping_text = "\n".join(
    [f"{', '.join(k)} -> {v}" for k, v in possible_events.items()]
)

all_possible_actions = [k for key_tuple in possible_events.keys() for k in key_tuple]
event_mapping_text = ", ".join(all_possible_actions)

def get_prompt(user_input):
    return f"""
<|im_start|>system
You are an assistant that converts noisy natural language into exactly ONE automation command — the action the user most likely intends to execute.

RULES:
1. Output exactly ONE command per input. Choose the PRIMARY intended action if multiple appear.
2. Do NOT invent commands. Only extract explicit imperative commands.
3. Ignore irrelevant text: small talk, emotions, explanations, questions, filler, hypotheticals.
4. Commands must start with recognized event keywords.
5. Do NOT try to guess names given by user, use them as they are, only guess actions or conditions if user was unclear about them.

EVENTS AND TARGETS:
- Mouse events ("click", "left click", "right click", "middle click", "double click", "open") → any string target.
- Click variable events:
    - "click all variable", "click every variable"... → all variables.
    - "click variable", "click var"... → top variable.
    - Must be followed up by a variable name.
- Image clicks ("click image", "click on image"...) → target image or text label.
- WRITE → text to type.
- PRESS → single key or key combination.
- SequenceEvent./START/STOP/PLAY recording → sequence name.
- LoopEvent./START/STOP → template variable if applicable.
- SLEEP → numeric or text duration.
- CAPTURE → no target.
- IF CASE ("if", "if case") → condition on text, image, or variable.  
    - Text: 'If text [optional colors] query exists', optional 'not' for negation.  
    - Image: 'If image query exists', optional 'not' for negation.  
    - Variable: 'If variable name matches condition'.
- END_IF ("end if", "stop if") → closes previous IF.
- SET VAR ("set var", "set variable", "make var", "make variable") → dynamic runtime variable.  
    - Syntax: name|type|desc  
    - type: "number" or "string"  
    - desc: numeric condition (e.g., ">10", "<50", "all") or string value  
    - Colors optional for numbers only. Strings must have a value.  
    - SET VAR is unrelated to templates.

TEMPLATES AND COLORS:
- Commands can include template variables (e.g., name) for looped or recorded sequences.
- Text color can be specified in targets (e.g., "click blue Submit") if relevant.
- Templates can only be used while recording sequences.
- Templates act as predefined variables for sequences, they are NOT SAME as runtime variables set by SET VAR event

OUTPUT FORMAT:
- Plain text, one command per line.
- No JSON, symbols, arrows, or commentary.
- Only extract the command literally requested by the user.

PRIMARY INTENT:
- If the user says multiple things, extract the most likely desired action.
- Ignore preparatory, meta, or warm-up instructions (like "focus up" or "listen").

<|im_end|>
<|im_start|>user
Convert this input into ONE valid automation command:
{user_input}
<|im_end|>
<|im_start|>assistant
"""
