import os
import wget
import json
import shutil
import unify
unify.activate("MarkingAssistant")

this_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(this_dir, "../data/dataset.json")

if not os.path.exists(dataset_path):
    wget.download(
        "https://github.com/unifyai/demos/raw/refs/heads/main/ai_tutor/data/dataset.json"
    )
    shutil.move("dataset.json", dataset_path)
with open(dataset_path, "r") as f:
    data = json.load(f)

# ensure the dataset upstream is the same as the local data
unify.Dataset(data, name="TestSet").sync()
