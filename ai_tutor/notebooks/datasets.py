import os
import wget
import json
import shutil
import argparse

import unify
unify.activate("MarkingAssistant", overwrite=True)

this_dir = os.path.dirname(os.path.abspath(__file__))

def save_dataset(name):
    dataset_path = os.path.join(this_dir, f"../data/{name}.json")

    if not os.path.exists(dataset_path):
        wget.download(
            f"https://github.com/unifyai/demos/raw/refs/heads/main/ai_tutor/data/{name}.json"
        )
        shutil.move(f"{name}.json", dataset_path)
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # ensure the dataset upstream is the same as the local data
    unify.Dataset(data, name=name.replace("_", " ").title().replace(" ", "")).sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    name = parser.parse_args().name
    assert name in ("students", "test_set")
    save_dataset(name)
