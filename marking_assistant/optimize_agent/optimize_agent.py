import os
import subprocess

this_dir = os.path.dirname(os.path.abspath(__file__))
for i in range(11):
    script_name = f"{this_dir}/iteration_{i}.py"
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)
    print(f"Finished {script_name}\n")
