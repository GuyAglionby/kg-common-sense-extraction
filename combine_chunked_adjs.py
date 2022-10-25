import argparse
import glob
import os
import pickle
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--splits", nargs="+", default=['train', 'test', 'dev'])
    args = parser.parse_args()

    for split in args.splits:
        if os.path.exists(f"{args.dir}/{split}.graph.adj.pk"):
            continue

        files = list(glob.glob(f"{args.dir}/{split}-*.graph.adj.pk"))
        if len(files) == 0:
            files = list(glob.glob(f"{args.dir}/{split}.graph.adj.pk*"))

        print(f"Found {len(files)} for {split}")
        files.sort(key=lambda x: int(x.split("/")[-1]
                                     .replace(f"{split}", "")
                                     .replace(".graph.adj.pk", "")
                                     .replace("-", "")))
        # this could be a regex (:
        file_offsets = [int(x.split("/")[-1]
                            .replace(f"{split}", "")
                            .replace(".graph.adj.pk", "")
                            .replace("-", "")) for x in files]

        offset_differences = [b - a for a, b in zip(file_offsets, file_offsets[1:])]
        if len(set(offset_differences)) != 1:
            problem_offset_values = [a[0] for a in list(Counter(offset_differences).most_common())[1:]]
            problem_offset_idxs = [i for b in problem_offset_values for i, a in enumerate(offset_differences) if a == b]
            for i in problem_offset_idxs:
                print(i + 1, files[i].split("/")[-1], files[i + 1].split("/")[-1])
            assert False

        full_adj = []
        for fname in files:
            with open(fname, 'rb') as f:
                full_adj.extend(pickle.load(f))

        print(f"Final len {len(full_adj)}")
        with open(f'{args.dir}/{split}.graph.adj.pk', 'wb') as f:
            pickle.dump(full_adj, f)


if __name__ == '__main__':
    main()
