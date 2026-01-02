from enum import Flag, auto
import os, json, re
from class_models.Context import Context
from core.state import RuntimeState
from models.VoiceTranscriber import VoiceTranscriber
from core.logging import get_logger
log = get_logger(__name__) 



class SequenceEvent(Flag):
    START   = auto()
    SAVE    = auto()
    PLAY    = auto()
    RESET   = auto()
    CLEAR_PREV = auto()


class SequenceHandler:
    def __init__(self, embd_func, filepath="sequences.json"):
        from utility import (cmp_txt_and_embs, extract_vars_from_contexts)
    
        self.voiceTranscriber = VoiceTranscriber()
        self.voiceTranscriber.listener_enabled = False
        self.embd_func = embd_func
        self.cmp_txt_and_embs = cmp_txt_and_embs
        self.extract_vars_from_contexts = extract_vars_from_contexts

        self.filepath = filepath
        if not os.path.exists(filepath):
            json.dump({}, open(filepath, "w"))
        self.key_embeds = []
        self.update_embds()


    
    def update_embds(self):
        data = json.load(open(self.filepath))
        self.data = data
        keys = [k for k in data.keys()]
        key_embeds = self.embd_func(keys)
        self.data_map = [(emb,key) for emb, key in zip(key_embeds, keys)]


    def strip_text(self, ctxs):
        for c in ctxs:
            if hasattr(c, "text") and isinstance(c.text, str):
                c.text = c.text.strip()
            if getattr(c, "sub_contexts", None):
                self.strip_text(c.sub_contexts)


    def save_sequence(self, events, name, vars_list=None):
        name = name.lower()
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        def remove_none(obj):
            if isinstance(obj, dict):
                return {
                    k: remove_none(v)
                    for k, v in obj.items()
                    if v not in (None, [], {})
                }
            elif isinstance(obj, list):
                return [remove_none(v) for v in obj if v not in (None, [], {})]
            return obj


        self.strip_text(events)

        data[name] = {
            "steps": [remove_none(e.to_dict()) for e in events],
            "vars": vars_list or []
        }

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)

        self.update_embds()






    def load_sequence(self, name, pause_listener_event=None):
        name = name.lower().strip()
        self.update_embds()

        matched = self.cmp_txt_and_embs(name, self.data_map, self.embd_func)
        sequence = self.data.get(matched["text"]) if matched else None
        if not sequence:
            return []

        # Gather variable values
        var_values = {}
        if sequence.get("vars") and pause_listener_event:
            pause_listener_event.clear()
            self.voiceTranscriber.listener_enabled = True
            for v in sequence["vars"]:
                var_name, var_type, predefined = v["name"], v.get("type","str"), v.get("value")
                if predefined:
                    if isinstance(predefined, list):
                        new_vals = []
                        for x in predefined:
                            if isinstance(x, str):
                                try:
                                    x_parsed = json.loads(x)
                                    new_vals.append(x_parsed)
                                except Exception:
                                    new_vals.append(x)
                            else:
                                new_vals.append(x)
                        var_values[var_name] = new_vals
                    else:
                        var_values[var_name] = [predefined]
                    continue
                else:
                    log.info(f"Please Enter the value for variable {var_name}")
                spoken = self.voiceTranscriber()
                var_values[var_name] = [x.strip() for x in (spoken.split(",") if var_type=="list" else [spoken])]
            pause_listener_event.set()
            self.voiceTranscriber.listener_enabled = False

        log.debug(var_values)

        # variable substitution
        VAR_PATTERN = re.compile(r"\{\{(\w+)(?:\.(\d+))?\}\}")
        def subst(text, current_vars=None):
            if not text:
                return text

            def repl(match):
                var = match.group(1)
                idx = match.group(2)

                if current_vars is None or var not in current_vars:
                    return match.group(0)  # leave as-is if variable missing

                value = current_vars[var]

                # scalar
                if not isinstance(value, list):
                    return str(value)

                # list or list-of-lists
                if idx is None:
                    first = value[0] if value else ""
                    if isinstance(first, list):
                        return str(first[0]) if first else ""
                    return str(first)
                else:
                    i = int(idx)
                    if isinstance(value[0], list):
                        return str(value[0][i]) if i < len(value[0]) else ""
                    else:
                        return str(value[i]) if i < len(value) else ""

            return VAR_PATTERN.sub(repl, text)
                
        def parse_loop(text):
            parts = text.lower().split()
            if "loop" not in parts:
                return (None, None)

            loop_idx = parts.index("loop")
            after = parts[loop_idx + 1:]

            if not after:
                return (None, None)

            if "as" in after:
                var = after[0]
                return ("template", var)
            return ("template", after[0])

        ctx_steps = [Context.from_dict(s).clone() for s in sequence["steps"]]
        final_steps = []

        for ctx in ctx_steps:
            new_ctx = ctx.clone()
            new_ctx.text = subst(new_ctx.text)

            loop_type, loop_val = parse_loop(ctx.text)

            if loop_type == "count":
                final_steps.append(ctx)
                continue

            if loop_type == "template":
                values = var_values.get(loop_val)

                if not values:
                    continue

                # build "other" current_vars for substitution (non-loop variables)
                other_vars = {k: v for k, v in var_values.items() if k != loop_val}

                if isinstance(values, list):
                    for val in values:  # val is current iteration
                        for sub in ctx.sub_contexts or []:
                            new_sub = sub.copy()
                            # combine loop variable + other variables
                            current_vars = {**other_vars, loop_val: val}
                            new_sub.text = subst(new_sub.text, current_vars)
                            final_steps.append(new_sub)
                else:
                    # scalar loop
                    for sub in ctx.sub_contexts or []:
                        new_sub = sub.copy()
                        current_vars = {**other_vars, loop_val: values}
                        new_sub.text = subst(new_sub.text, current_vars)
                        final_steps.append(new_sub)
                continue

            # normal step
            final_steps.append(ctx)
        return final_steps
    
    



    def process_sequence_event(self, rs: RuntimeState, voiceTranscriber: VoiceTranscriber):
        
        if not rs.recording_stack or not rs.recording_state:
            rs.recording_state = {"active": False, "contexts": [], "name": ""}
            rs.recording_stack = [rs.recording_state["contexts"]]
        
        if rs.action_event == SequenceEvent.START:
            rs.recording_state["active"] = True
            rs.recording_state["contexts"] = []
            rs.recording_state["name"] = rs.target_text
            rs.recording_stack = [rs.recording_state["contexts"]]

        elif rs.action_event == SequenceEvent.SAVE:
            vars_list = self.extract_vars_from_contexts(rs.recording_state["contexts"])
            self.save_sequence(rs.recording_state["contexts"], rs.recording_state.get("name"), vars_list)
            rs.recording_state.update({"active": False, "contexts": [], "name": ""})

        elif rs.action_event == SequenceEvent.PLAY:
            voiceTranscriber.listener_enabled = False
            loaded_sequence = self.load_sequence(
                                    name=rs.target_text, 
                                    pause_listener_event=voiceTranscriber.pause_listener_event
                                )
            if loaded_sequence:
                rs.context_queue.extend(loaded_sequence)
            voiceTranscriber.listener_enabled = True

        elif rs.action_event == SequenceEvent.RESET and rs.recording_state["active"]:
            rs.recording_state["contexts"] = []
            rs.recording_stack = [rs.recording_state["contexts"]]

        elif rs.action_event == SequenceEvent.CLEAR_PREV and rs.recording_state["active"]:
            if rs.recording_stack[-1]:
                last_item = rs.recording_stack[-1][-1]
                log.info(f"Cleared step '{last_item.text if hasattr(last_item, 'text') else last_item}'")
                rs.recording_stack[-1].pop()


        return False

