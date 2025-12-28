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
                    var_values[var_name] = [str(x).strip() for x in (predefined if isinstance(predefined,list) else [predefined])]
                    continue
                spoken = self.voiceTranscriber()
                var_values[var_name] = [x.strip() for x in (spoken.split(",") if var_type=="list" else [spoken])]
            pause_listener_event.set()
            self.voiceTranscriber.listener_enabled = False

        # variable substitution
        def subst(text):
            if not text: return ""
            for var, vals in var_values.items():
                if vals: text = text.replace(f"{{{{{var}}}}}", vals[0])
            return text
        
        def parse_loop(text):
            parts = text.lower().split()
            if len(parts) >= 2 and parts[0] == "loop":
                # numeric loop
                if parts[1].isdigit():
                    return ("count", int(parts[1]))
                # explicit template
                if len(parts) >= 4 and parts[-2] == "as" and (parts[-1] == "template" or parts[-1] == "variable"):
                    return ("template", parts[1])
                # implicit template: "loop varname"
                return ("template", parts[1])
            return (None, None)

        ctx_steps = [Context.from_dict(s) for s in sequence["steps"]]
        final_steps = []

        for ctx in ctx_steps:
            ctx.text = subst(ctx.text)

            loop_type, loop_val = parse_loop(ctx.text)

            # NUMERIC LOOP
            if loop_type == "count":
                final_steps.append(ctx)
                continue

            # TEMPLATE LOOP
            if loop_type == "template":
                values = var_values.get(loop_val)

                if not values: # no variable, skip
                    continue

                # expand
                for val in values:
                    for sub in ctx.sub_contexts or []:
                        new_sub = sub.copy()
                        new_sub.text = new_sub.text.replace(f"{{{{{loop_val}}}}}", val)
                        final_steps.append(new_sub)
                continue

            # NORMAL STEP 
            final_steps.append(ctx)

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
            print("loaded sequence: ", loaded_sequence)
            print("a:", len(context_queue))
            if loaded_sequence:
                context_queue.extend(loaded_sequence)
            print("b:", len(context_queue))
            voiceTranscriber.listener_enabled = True


        return False, recording_state, recording_stack

