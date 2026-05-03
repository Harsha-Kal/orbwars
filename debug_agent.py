import json, os

log_dir = r'c:\Users\Sreeh\OneDrive\Desktop\orbwars\agent logs'
files = sorted(os.listdir(log_dir))

for fname in files:
    path = os.path.join(log_dir, fname)
    with open(path) as f:
        data = json.load(f)

    durations = [entry['duration'] for step in data for entry in step]
    errors = [(i, entry['stderr']) for i, step in enumerate(data)
              for entry in step if entry.get('stderr')]
    outputs = [(i, entry['stdout']) for i, step in enumerate(data)
               for entry in step if entry.get('stdout')]

    print(fname)
    print("  Steps:", len(data))
    print("  Duration max=%.2fms avg=%.2fms" % (max(durations)*1000, sum(durations)/len(durations)*1000))
    if errors:
        for i, e in errors[:3]:
            print("  STDERR step %d:" % i, e[:300])
    if outputs:
        for i, o in outputs[:3]:
            print("  STDOUT step %d:" % i, o[:300])
    if not errors and not outputs:
        print("  (no stdout/stderr — agent ran silently)")
    print()
