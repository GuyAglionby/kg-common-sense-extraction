import argparse
import glob

import jsonlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-dir")
    parser.add_argument("--results-dir")
    args = parser.parse_args()
#
    for split in ['train', 'test', 'dev']:
        res_f = list(glob.glob(f'{args.results_dir}/{split}_emb_grounded.jsonl'))[0]
        grounded_f = list(glob.glob(f'{args.grounded_dir}/{split}.grounded.jsonl'))[0]
        with jsonlines.open(grounded_f) as f:
            grounded = list(f)
        with jsonlines.open(res_f) as f:
            res = list(f)

        n_concepts = len(res[0]['answers'])

        grounded = [grounded[i:i+n_concepts] for i in range(0, len(grounded), n_concepts)]
        new_grounded = []
        for g, r in zip(grounded, res):
            question = [x[0] for x in r['question']]
            for i, gg in enumerate(g):
                gg['ac'] = [x[0] for x in r['answers'][str(i)]]
                gg['qc'] = [c for c in question if c not in gg['ac']]
                new_grounded.append(gg)
        with jsonlines.open(f"{args.results_dir}/{split}.grounded.jsonl", 'w') as f:
            f.write_all(new_grounded)


if __name__ == '__main__':
    main()
