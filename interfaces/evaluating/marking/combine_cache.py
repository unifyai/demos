import os
import json
this_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_dir, ".cache0.json"), "r") as f:
    cache0 = json.load(f)
with open(os.path.join(this_dir, ".cache1.json"), "r") as f:
    cache1 = json.load(f)
cache = {**cache0, **cache1}
assert len(cache) == 122
with open(os.path.join(this_dir, ".cache.json"), "w") as f:
    f.write(json.dumps(cache))
