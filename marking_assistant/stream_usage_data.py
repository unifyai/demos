import json
import os
import random
from datetime import datetime, timedelta, timezone

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

unify.initialize_async_logger()

while True:
    sample = random.choice(usage_data)
    sample["student/timestamp"] = (
        datetime.now(timezone.utc)
        + timedelta(
            seconds=random.randint(-90, 90),
        )
    ).isoformat()
    unify.log(**sample)
