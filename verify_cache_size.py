import json
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_dir, ".cache.json")) as file:
    data = json.load(file)
print("length of .cache.json:", len(data))
