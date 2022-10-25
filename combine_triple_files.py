import argparse
import glob
import json
from collections import Counter

import jsonlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stubs", nargs="+", help="stub for different files")
    args = parser.parse_args()

    for stub in args.stubs:
        concepts_files = list(glob.glob(f'{stub}-*-concepts.jsonl'))
        concepts = []
        for filename in concepts_files:
            with jsonlines.open(filename) as f:
                concepts.extend(list(f))

        line_idxs_files = list(glob.glob(f'{stub}-*-line_idxs.txt'))
        line_idxs = []
        for filename in line_idxs_files:
            with open(filename) as f:
                line_idxs.extend([int(a) for a in f.read().split("\n") if len(a.strip())])

        path_lengths_files = list(glob.glob(f'{stub}-*-path-lens.json'))
        path_lengths = Counter()
        for filename in path_lengths_files:
            with open(filename) as f:
                l = json.load(f)
                path_lengths += Counter(l)

        concepts_ordered = list(zip(concepts, line_idxs))
        concepts_ordered.sort(key=lambda x: x[1])
        concepts_ordered, idxs_ordered = zip(*concepts_ordered)

        with jsonlines.open(f"{stub}-all-concepts.jsonl", 'w') as f:
            f.write_all(concepts_ordered)

        with open(f'{stub}-all-path-lens.json', 'w') as f:
            json.dump(dict(path_lengths), f)


if __name__ == '__main__':
    main()