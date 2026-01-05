import re

# Finding which color/s is/are mentioned in context
colors = ["black","white","red","green","blue","yellow","orange","brown","gray","purple"]
color_pattern = r"\b(" + "|".join(colors) + r")\b"
def find_colors(ctx):
    matches = []
    colors = []

    for m in re.finditer(color_pattern, ctx, re.I):
        # Check if there's non-space text after this color
        if re.search(r"\S", ctx[m.end():]):
            matches.append(m)
            colors.append(m.group(0).lower())

    if matches:
        last_match = max(matches, key=lambda m: m.end())
        ctx = ctx[last_match.end():].strip()
        ctx = re.sub(r"^[\s\W]+", "", ctx)

    return colors or None, ctx


print(find_colors("click blue or red youtube"))