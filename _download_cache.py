import os

import unify
import wget

if os.path.exists(".cache.json"):
    os.remove(".cache.json")
if os.path.exists(".latest_cache.json"):
    os.remove(".latest_cache.json")
if os.path.exists(".prev_cache.json"):
    os.remove(".prev_cache.json")

wget.download(
    "https://raw.githubusercontent.com/unifyai/demos/1cd42e27931037b69ea87821588dc097e1af68be/marking_assistant/.cache.json",
)
os.rename(".cache.json", ".latest_cache.json")

wget.download(
    "https://raw.githubusercontent.com/unifyai/demos/ad1208c0fd9b60f6a2c19123b416757f327a5686/marking_assistant/.cache.json",
)
os.rename(".cache.json", ".prev_cache.json")

unify.cache_file_intersection(
    ".latest_cache.json",
    ".prev_cache.json",
    ".intersection_cache.json",
)
unify.subtract_cache_files(
    ".latest_cache.json",
    ".intersection_cache.json",
    ".cache.json",
)
