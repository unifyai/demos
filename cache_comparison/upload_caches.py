import json
import os
import shutil

import unify
import wget

unify.activate("CacheComparison", overwrite=True)

cache_versions = {
    "added .cache.json for the agent improvement notebook.": "822c7502910e21ceeaaf0c4246df7213baff25d8",
    "updated .cache.json up to second iteration.": "7962af859664b869353933d79714dc0002414da2",
    "updated .cache.json up to third iteration, add_marking_guidelines": "27c13fafcffe4c10bf677c5ddbd54ab41168f630",
    "updated .cache.json up to fourth iteration, add_structured_output": "2bd6577181939c98e77ca786a49e91d936b057a0",
    "updated .cache.json up to fifth iteration, align_context": "bb78c0785fcd6f4ffac8211ebb14839399eeb104",
    "updated .cache.json up to sixth iteration.": "f3e5db146ddea148dc2f1284c1920790c2339aa0",
    "updated .cache.json up to seventh iteration, mark_type_reasoning.": "98668567a72fcc852b76495a8230c587bbbd17d0",
    "updated .cache.json up to eighth iteration, queries_per_subquestion.": "e1ee8f4f42ef99296fe105310430612f714349eb",
    "fixed .cache.json for eighth iteration, queries_per_subquestion.": "8954881616fe3a115648e0f5362b106c30c22b75",
    "reverted .cache.json back to SIXTH (not seventh) iteraton, mark_type_reasoning.": "8e249391b921bbc87fdac70407931cfb695dabc4",
    "updated .cache.json to iteration seven, queries_per_subquestion.": "beaf975e2c7da2bd09f5fa4e1fd4f66474f20d0c",
    "updated .cache.json to iteration eight, with_preceeding_context": "7191aa6b5752431b1e8336177dbc5ed7dcc6e21d",
    "updated .cache.json to iteration nine, queries_per_mark": "07cf944193e134ba9ae6bfa70ceed60d0d791b65",
}

caches = dict()

for commit_msg, commit_hash in cache_versions.items():
    new_fname = f".cache_{commit_hash}.json"
    if not os.path.exists(new_fname):
        wget.download(
            f"https://github.com/unifyai/demos/raw/{commit_hash}/marking_assistant/.cache.json",
        )
        shutil.move(".cache.json", new_fname)
    with open(new_fname, "r") as f:
        caches[commit_msg] = json.load(f)

cache_data = list()
for commit_msg, cache in caches.items():
    for inp, out in cache.items():
        if isinstance(out, str):
            out = json.loads(out)

        input_fn = inp.split("{")[0]

        input_dict = "{" + "{".join(inp.split("{")[1:])
        input_dict = "}".join(input_dict.split("}")[:-1]) + "}"
        input_dict = json.loads(input_dict)

        output_type = "types" if inp.split("}")[-1] == "_res_types" else "response"

        cache_data.append(
            {
                "commit_msg": commit_msg,
                "commit_hash": cache_versions[commit_msg],
                "function": input_fn,
                "input": input_dict,
                "output": out,
                "output_type": output_type,
            },
        )

unify.create_logs(entries=cache_data)
