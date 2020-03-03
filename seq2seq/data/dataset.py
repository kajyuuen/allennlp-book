#!/usr/bin/env python3

import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--output-train", type=str, default="./data/train.tsv")
    parser.add_argument("--output-valid", type=str, default="./data/valid.tsv")
    parser.add_argument("--random-seed", type=int, default=1)
    args = parser.parse_args()

    assert args.valid_size < args.max_value ** 2

    random.seed(args.random_seed)

    dataset = [
       f"{x} + {y}\t{x + y}"
       for x in range(args.max_value)
       for y in range(args.max_value)
    ]
    random.shuffle(dataset)

    trains = dataset[args.valid_size:]
    valids = dataset[:args.valid_size]

    with open(args.output_train, "w") as f:
        f.write("\n".join(trains))

    with open(args.output_valid, "w") as f:
        f.write("\n".join(valids))


if __name__ == "__main__":
    main()
