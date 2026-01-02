import re
from class_models.Context import Context

# Extracts variables from inside {{}} brackets (aka templates)
# so they could be replaces with real usable values before inference
def extract_vars_from_contexts(contexts):
    vars_dict = {}
    visited = set()

    def add(name, typ):
        vars_dict.setdefault(name, typ)

    def walk(ctxs, loop_vars=None):
        loop_vars = loop_vars or set()

        for ctx in ctxs:
            if isinstance(ctx, Context):
                if id(ctx) in visited:
                    continue
                visited.add(id(ctx))

                new_loop_vars = loop_vars.copy()

                if ctx.text:
                    t = ctx.text.strip().lower()

                    # loop detection
                    if t.startswith(("loop", "start loop")) and "as template" in t:
                        var = t.split("as template")[0] \
                            .replace("start loop", "") \
                            .replace("loop", "") \
                            .strip()
                        add(var, "list")
                        new_loop_vars.add(var)

                    # normal template vars
                    for v in re.findall(r"\{\{\s*(.*?)\s*\}\}", ctx.text):
                        if v not in new_loop_vars:
                            add(v, "str")

                if ctx.sub_contexts:
                    walk(ctx.sub_contexts, new_loop_vars)

            elif isinstance(ctx, list):
                walk(ctx, loop_vars)

    walk(contexts)
    return [{"name": k, "type": v} for k, v in vars_dict.items()]