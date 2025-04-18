import argparse
import os
import subprocess

import unify

unify.set_context("Evals", overwrite=True)

this_dir = os.path.dirname(os.path.abspath(__file__))

args = argparse.ArgumentParser()
args.add_argument("--iteration", type=int, default=11)
args = args.parse_args()

for i in range(args.iteration):
    script_name = f"{this_dir}/iteration_{i}.py"
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)
    print(f"Finished {script_name}\n")
