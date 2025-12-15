import json

path = "datasets/domain_06b/domain_06b_dataset.jsonl"

# Read all lines
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

seen = set()
unique = []

for line in lines:
    try:
        obj = json.loads(line)
    except:
        continue

    key = (obj["user_input"].strip(), obj["assistant_output"].strip())
    if key not in seen:
        seen.add(key)
        unique.append(obj)

# Overwrite same file
with open(path, "w", encoding="utf-8") as f:
    for obj in unique:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("done")
