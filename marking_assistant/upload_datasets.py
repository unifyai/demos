import json
import os

import unify
import wget

unify.activate("MarkingAssistant")
for dataset in unify.list_datasets():
    unify.delete_dataset(dataset)

# Users

if os.path.exists("users.json"):
    os.remove("users.json")

wget.download(
    "https://github.com/unifyai/demos/"
    "raw/refs/heads/main/marking_assistant/"
    "data/users.json",
)

with open("users.json", "r") as f:
    users = json.load(f)

users_dataset = unify.Dataset(users, name="Users")
users_dataset.sync()

# Test Set

if os.path.exists("test_set.json"):
    os.remove("test_set.json")

wget.download(
    "https://github.com/unifyai/demos/"
    "raw/refs/heads/main/marking_assistant/"
    "data/test_set.json",
)

with open("test_set.json", "r") as f:
    test_set = json.load(f)

test_set = unify.Dataset(test_set, name="TestSet")
test_set.sync()

# Sub Test Sets

for size in [10, 20, 40, 80, 160]:
    test_set[0:size].set_name(f"TestSet{size}").sync()
