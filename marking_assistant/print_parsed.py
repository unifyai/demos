import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path of the parsed .json file to print", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(this_dir, args.path)
    with open(pdf_path) as f:
        parsed = json.load(f)
    for _, dct in parsed.items():
        print("\n" + dct["text"])
