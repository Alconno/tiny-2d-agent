import os

file_sizes = []

for root, dirs, files in os.walk('.'):
    for f in files:
        path = os.path.join(root, f)
        try:
            size = os.path.getsize(path)
            file_sizes.append((size, path))
        except OSError:
            pass  # skip unreadable files

# Sort by size descending
file_sizes.sort(reverse=True)

for size, path in file_sizes:
    print(f"{size:>10} bytes  {path}")
