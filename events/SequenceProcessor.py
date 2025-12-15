from enum import Flag, auto
import os, json, re
from class_models.Context import Context

class SequenceEvent(Flag):
    START   = auto()
    SAVE    = auto()
    PLAY    = auto()


class SequenceProcessor:
    def __init__(self, embd_func, filepath="sequences.json"):
        from VoiceTranscriber import VoiceTranscriber
        from utility.dogshitretard import cmp_txt_and_embs, extract_vars_from_steps

        self.voiceTranscriber = VoiceTranscriber()
        self.voiceTranscriber.listener_enabled = False
        self.embd_func = embd_func
        self.cmp_txt_and_embs = cmp_txt_and_embs
        self.extract_vars_from_steps_func = extract_vars_from_steps

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



    def save_sequence(self, events, name, vars_list=None):
        name = name.lower()
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        data[name] = {
            "steps": [e.to_dict() for e in events],
            "vars": vars_list or []
        }

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)

        self.update_embds()




    def load_sequence(self, name, pause_listener_event=None):
        name = name.lower().strip()
        self.update_embds()

        matched = self.cmp_txt_and_embs(name, self.data_map, self.embd_func)
        sequence = self.data.get(matched["text"], None) if matched else None
        if not sequence: return []

        # Extract vals
        vars_list = sequence.get("vars", [])
        var_values = {}
        self.voiceTranscriber.listener_enabled = True
        if vars_list and pause_listener_event:
            pause_listener_event.clear()
            for var_dict in vars_list:
                var_name = var_dict["name"]
                var_type = var_dict.get("type", "str")
                predefined = var_dict.get("value", None)

                if predefined:
                    if isinstance(predefined, list):
                        var_values[var_name] = [str(v).strip() for v in predefined]
                    else:
                        var_values[var_name] = [str(predefined).strip()]
                    continue

                print(f"Provide value(s) for '{var_name}'")
                spoken = self.voiceTranscriber()
                if var_type == "list":
                    var_values[var_name] = [v.strip() for v in spoken.split(",")]
                else:
                    var_values[var_name] = [spoken.strip()]
            pause_listener_event.set()

        self.voiceTranscriber.listener_enabled = False

        def subst(text: str, override=None):
            if not text:
                return ""
            if override is not None:
                text = text.replace(f"{{{{{override[0]}}}}}", override[1])
            else:
                for var, vals in var_values.items():
                    if vals:
                        text = text.replace(f"{{{{{var}}}}}", vals[0])
            return text

        def expand(ctx: Context, override=None):
            ctx = ctx.copy()
            ctx.text = subst(ctx.text, override)
            if "loop" in ctx.meta:
                loop_var = ctx.meta["loop"].split(" as ")[0]
                vals = var_values.get(loop_var, [])
                out = []
                for v in vals:
                    loop_override = (loop_var, v)
                    for c in ctx.sub_contexts or []:
                        out.extend(expand(c, loop_override))
                return out
            if "loop_count" in ctx.meta:
                out = []
                for _ in range(ctx.meta["loop_count"]):
                    for c in ctx.sub_contexts or []:
                        out.extend(expand(c, override))
                return out
            if ctx.sub_contexts:
                new_children = []
                for c in ctx.sub_contexts:
                    new_children.extend(expand(c, override))
                ctx.sub_contexts = new_children
            return [ctx]

        ctx_steps = [Context.from_dict(s) for s in sequence["steps"]]
        final_steps = []
        for step in ctx_steps:
            final_steps.extend(expand(step, None))
        return final_steps



    def process_sequence_event(self, action_result, target_text,  
                               recording_state, recording_stack,
                               voiceTranscriber, pause_listener_event, context_queue):
        
        if action_result == SequenceEvent.START:
            recording_state["active"] = True
            recording_state["contexts"] = []
            recording_state["name"] = target_text
            recording_stack = [recording_state["contexts"]]

        elif action_result == SequenceEvent.SAVE:
            vars_list = self.extract_vars_from_steps_func(recording_state["contexts"])
            self.save_sequence(recording_state["contexts"], recording_state.get("name"), vars_list)
            recording_state.update({"active": False, "contexts": [], "name": ""})

        elif action_result == SequenceEvent.PLAY:
            voiceTranscriber.listener_enabled = False
            loaded_sequence = self.load_sequence(name=target_text, pause_listener_event=pause_listener_event)
            if loaded_sequence:
                context_queue.extend(loaded_sequence)
            voiceTranscriber.listener_enabled = True


        return False, recording_state, recording_stack




"""
                    {"text": "click below of Name"},
                    {"text": "Write eternal fire"},
                    {"text": "Press enter"},
                    {"text": "Wait 1"},
                    {"text": "Focus 831, 242, 903, 650"},
                    {"text": "set variable items|number|yellow <45"},
                    {"text": "click top variable items"}
"""