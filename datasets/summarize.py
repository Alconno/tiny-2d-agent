import json
from pathlib import Path

base_dir = Path(r"C:\USERS\OPA\DESKTOP\PYTORCH1\WEB-AI-HELPER\DATASETS\domain_06b")
output_file = base_dir / "domain_06b_dataset.jsonl"
events_dir = base_dir / "Events"

jsonl_files = [f for f in events_dir.glob("*.jsonl") if f.name != "domain_06b_dataset.jsonl"]

with open(output_file, "w", encoding="utf-8") as out_f:
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip().rstrip(',')  # remove trailing commas
                if line.startswith("//") or not line:
                    continue
                try:
                    data = json.loads(line)
                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file_path}: {line[:50]}...")

print(f"All data extracted to {output_file}")
