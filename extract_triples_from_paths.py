import argparse
import glob
import itertools
import json
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import jsonlines
import networkx as nx
from tqdm import tqdm

from conceptnet_utils import init_cpnet, concept_to_id, cpnet_simple_has_node, get_cpnet_simple


def get_next_path_length(gen):
    try:
        path = next(gen)
        return itertools.chain([path], gen), len(path)
    except nx.NetworkXNoPath:
        return gen, -1
    except StopIteration:
        return gen, -1


def get_interim_nodes_via_pathfinding(max_triple_num, max_k, data):
    line_idx, qc_ids, ac_ids = data
    triples = set()
    generators = []
    for qid in qc_ids:
        for aid in ac_ids:
            # no reason why qid or aid not in cpnet_simple.nodes, not sure why this is here - ga384

            if qid == aid or not cpnet_simple_has_node(qid) or not cpnet_simple_has_node(aid):
                continue
            gen = nx.shortest_simple_paths(get_cpnet_simple(), source=qid, target=aid)
            generators.append(gen)
            # an attempt to order generators by the length of path they generate, rather than cycling through them
            # gen, path_len = get_next_path_length(gen)
            # if path_len >= 0:
            #     generators.append((gen, path_len))

    # current_path_len = min(a[1] for a in generators)

    path_lens = Counter()
    while len(generators) and len(triples) < max_triple_num:
        # gen, path_len = generators.pop(0)
        # if path_len < 0:
        #     continue
        # if path_len > current_path_len:
        #     generators.append((gen, path_len))
        gen = generators.pop(0)
        try:
            path = next(gen)
            # number of elements - 1 = path length
            if len(path) - 1 > max_k:
                continue
            # each hop's order doesn't matter - we always add backwards links
            hops = {tuple(sorted(hop)) for hop in zip(path, path[1:])}
            before = len(triples)
            triples |= hops
            after = len(triples)
            if before != after:
                path_lens[len(path)] += 1
            # generators.append(get_next_path_length(gen))
            generators.append(gen)
        except nx.NetworkXNoPath:
            continue
        except StopIteration:
            continue

    return str(line_idx), list(triples), path_lens


def handle_file(filename, output_dir, num_processes):
    with open(filename) as f:
        lines = list(f)
    qa_data = []
    for i, line in enumerate(lines):
        dic = json.loads(line)
        q_ids = set(concept_to_id(c) for c in dic['qc'])
        a_ids = set(concept_to_id(c) for c in dic['ac'])
        q_ids = q_ids - a_ids
        qa_data.append((i, q_ids, a_ids))

    max_edge_num = 10000
    max_k = 8

    print(f"{len(qa_data)=}")

    n_per_split = len(qa_data) // 2

    out_fname = filename.split("/")[-1]
    output_stub = f"{output_dir}/{out_fname}-".replace("//", "/")

    max_i = max([-1] + [int(a.replace("-line_idxs.txt", "").replace(output_stub, ""))
                        for a in glob.glob(f"{output_stub}*-line_idxs.txt")])

    done_lines = set()
    for fname in glob.glob(f"{output_stub}*-line_idxs.txt"):
        with open(fname) as f:
            done_lines |= set([int(a) for a in f.read().split("\n") if len(a.split())])

    qa_data = [a for a in qa_data if a[0] not in done_lines]

    with Pool(num_processes) as p:
        # Here, just generate the candidates. we'll coerce everything into a graph later
        par = partial(get_interim_nodes_via_pathfinding, max_edge_num, max_k)
        for i, j in tqdm(list(enumerate(range(0, len(qa_data), n_per_split), max_i + 1)), desc="Splits"):
            this_split_filename = f"{output_stub}{i}"
            if os.path.exists(f"{this_split_filename}-concepts.jsonl"):
                print(f"Split exists: {this_split_filename}")
                continue
            this_split_data = qa_data[j:j + n_per_split]

            it = p.imap(par, this_split_data)
            res = list(tqdm(it, total=len(this_split_data), desc="qa pairs", smoothing=0))

            line_idxs, triples, path_lens = zip(*res)
            c = Counter()
            for pl in path_lens:
                c += pl

            with jsonlines.open(f"{this_split_filename}-concepts.jsonl", 'w') as fout:
                fout.write_all(triples)
            with open(f"{this_split_filename}-line_idxs.txt", 'w') as fout:
                fout.write("\n".join(line_idxs))
            with open(f"{this_split_filename}-path-lens.json", 'w') as fout:
                json.dump(dict(c), fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="qc/ac files for candidates")
    parser.add_argument("--kg", help="kg directory")
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--output_dir", default="./", type=str)
    args = parser.parse_args()

    print("loading resources")
    init_cpnet(args.kg, simple=True)

    print("iterating")

    for filename in tqdm(args.files, desc="files"):
        handle_file(filename, args.output_dir, args.num_processes)

if __name__ == '__main__':
    main()