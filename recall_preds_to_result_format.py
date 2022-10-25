import os

import argparse
import pickle

import jsonlines

from utils import name_to_split


def get_other_file(f):
    if 'scores' in f:
        return f.replace('scores', 'sros').replace('pkl', 'jsonl')
    else:
        return f.replace('sros', 'scores').replace('jsonl', 'pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+')
    parser.add_argument("--output-dir", default='.')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for fname in args.files:
        split = name_to_split(fname)
        if 'scores' in fname:
            scores = fname
            sros = get_other_file(fname)
        else:
            scores = get_other_file(fname)
            sros = fname

        with jsonlines.open(sros) as f:
            sros = list(f)

        with open(scores, 'rb') as f:
            scores = pickle.load(f)

        out = []
        for q_score, q_sro in zip(scores, sros):
            indices = q_score.argsort().tolist()
            sorted_score = q_score[indices[:2000]]
            sorted_sro = [q_sro[i] for i in indices[:2000]]
            out.append((sorted_sro, sorted_score))

        with open(f"{args.output_dir}/{split}_recall_result.pkl", "wb") as f:
            pickle.dump(out, f)

if __name__ == '__main__':
    main()
