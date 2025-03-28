import json
import os

import unify
import wget

unify.activate("MarkingAssistant")

unify.set_context("Usage")

if not os.path.exists("usage_data.json"):
    wget.download(
        "https://github.com/unifyai/demos/"
        "raw/refs/heads/main/marking_assistant/"
        "data/usage_data.json",
    )

with open("usage_data.json", "r") as f:
    usage_data = json.load(f)

unify.create_logs(entries=usage_data)
