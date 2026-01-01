from core.state import RuntimeState

def apply_gpt_to_context(rs: RuntimeState):
    ctx = rs.current_context
    if not ctx or not ctx.text:
        return ""

    text = ctx.text.strip()
    if not rs.models.gpt_func:
        return text

    res = rs.models.gpt_func(text)
    if not res:
        return text

    lines = [l.strip() for l in res.split("\n") if l.strip()]
    if not lines:
        return ""

    rs.context_queue.extend(lines[1:])
    return lines[0]
