import os
import wget
import json
import shutil

import unify
unify.activate("MarkingAssistant", overwrite=True)

this_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(this_dir, "../data/usage_data.json")

if not os.path.exists(data_path):
    wget.download(
        "https://github.com/unifyai/demos/raw/refs/heads/main/ai_tutor/data/usage_data.json"
    )
    shutil.move("usage_data.json", data_path)
with open(data_path, "r") as f:
    data = json.load(f)

unify.create_logs(context="Usage", entries=data)
