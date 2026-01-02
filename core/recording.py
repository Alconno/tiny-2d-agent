from core.state import RuntimeState

def append_to_recording_seq(rs: RuntimeState):
    section = rs.recording_stack[-1]
    if rs.is_template:
        color = "" if rs.color_list is None else rs.color_list.group(1).strip()
        var_name = rs.is_template.group(1).replace(color, "").strip()
        rs.current_context.text = f"{rs.action_result.get('span','Type').strip()} {color} {{{{{var_name}}}}}"
    if rs.current_context.text:
        rs.current_context.text = rs.current_context.text.strip()
    section.append(rs.current_context)
    return False # success

def append_condition_to_recording_seq(rs: RuntimeState):
    if rs.recording_state["active"]:
        rs.current_context.sub_contexts = []
        parent = rs.recording_stack[-1]
        parent.append(rs.current_context)
        rs.recording_stack.append(rs.current_context.sub_contexts)
    return False # success
