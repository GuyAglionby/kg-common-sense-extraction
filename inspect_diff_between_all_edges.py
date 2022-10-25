import argparse
import pickle

import jsonlines


def load_adj_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    out = []
    for question in data:
        if isinstance(question, tuple):
            rel_adj, nodes, _, _ = question
        elif isinstance(question, dict):
            rel_adj = question['adj']
            nodes = question['concepts']
        else:
            raise ValueError(f"{type(question)}")
        n_nodes = rel_adj.shape[1]
        rel_adj_flat = rel_adj.toarray().reshape((-1, n_nodes, n_nodes)).nonzero()
        relations = rel_adj_flat[0].tolist()
        sources = nodes[rel_adj_flat[1]].tolist()
        targets = nodes[rel_adj_flat[2]].tolist()

        rel_list = set(zip(sources, relations, targets))
        out.append({
            "nodes": set(nodes.tolist()),
            "relations": rel_list
        })
    return out


def get_answer_choice(statement, i):
    i = 'ABCD'[i]
    for x in statement['question']['choices']:
        if x['label'] == i:
            return x['text']
    assert False


def rel_to_txt(sros, vocab, relations):
    return [f"({vocab[s]}, {relations[r]}, {vocab[o]})" for s, r, o in sros]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statement-path")
    parser.add_argument("--cpnet-dir")
    parser.add_argument("--adj-with-all")
    parser.add_argument("--adj-with-pathfinding")
    args = parser.parse_args()

    with_all = load_adj_pickle(args.adj_with_all)
    with_pathfinding = load_adj_pickle(args.adj_with_pathfinding)

    with open(f"{args.cpnet_dir}/entity_vocab.txt") as f:
        vocab = [a for a in f.read().split("\n") if len(a)]

    with open(f"{args.cpnet_dir}/relations.tsv") as f:
        relations = [a.split("\t")[1] for a in f.read().split("\n") if len(a)]

    with jsonlines.open(args.statement_path) as f:
        statements = list(f)

    for i, (w_all, w_path) in enumerate(zip(with_all, with_pathfinding)):
        # hardcoded for 4 answer choices
        statement = statements[i // 4]
        print(f"Q: {statement['question']['stem']}")
        print(f"A: {get_answer_choice(statement, i % 4)}")
        print()
        print("Intersection")
        print("\n".join(rel_to_txt(w_path['relations'].intersection(w_all['relations']), vocab, relations)))
        print()
        print("Scenario ALL exclusive edges")
        print("\n".join(rel_to_txt(w_all['relations'] - w_path['relations'], vocab, relations)))
        print()
        print("Scenario PATH exclusive edges")
        print("\n".join(rel_to_txt(w_path['relations'] - w_all['relations'], vocab, relations)))
        print("\n\n")

        if i == 4:
            break


if __name__ == '__main__':
    main()
