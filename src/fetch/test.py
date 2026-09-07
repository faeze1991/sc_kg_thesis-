print("Script started")

with open("data/raw/pdc2020c.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(f"Line {i}: {line[:80]}")
        if i >= 4:
            break

print("Script finished")