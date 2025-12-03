import json
import sys


def check(filename: str):
    with open(filename, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    for model, metric_data in data.items():
        for metric, scores in metric_data.items():
            if len(scores) != 1000:
                print(f"{model}|{metric}|{len(scores)} scores missing")

            zeroes = [_ for _ in scores if _ == 0.0]
            if len(zeroes) > 0:
                print(f"{model}|{metric}|{len(zeroes)} scores = 0")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check.py detailed_scores_per_model.json")
        exit(1)

    check(sys.argv[1])
