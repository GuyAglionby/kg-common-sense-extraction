import argparse
import os

import torch
from tqdm import tqdm

from utils import name_to_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", choices=['sbert', 'deepblue'])
    parser.add_argument("--chunksize", default=39)
    parser.add_argument("--output-parent-dir", required=True)
    parser.add_argument("--split", required=True, choices=['test', 'train', 'dev'])
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    output_subdir = f'per_q_edge_scores_{args.encoding}_{args.dataset}'
    output_dir = f'{args.output_parent_dir}/{output_subdir}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    question_encoding_path = f'data/{args.dataset}_{args.split}_{args.encoding}_question_encodings.pt'
    question_embs = torch.load(question_encoding_path)
    fact_embs = torch.load(f"{args.encoding}_embedding_embs.pt")
    split = name_to_split(question_encoding_path)

    chunk_start_idxs = list(range(0, question_embs.shape[0], args.chunksize))

    for start_idx in tqdm(chunk_start_idxs):
        chunk_file = f"{output_dir}/{split}_{start_idx}.pt"
        if os.path.exists(chunk_file):
            continue
        scores = torch.cdist(question_embs[start_idx:start_idx + args.chunksize], fact_embs)
        torch.save(scores, chunk_file)


if __name__ == '__main__':
    main()
