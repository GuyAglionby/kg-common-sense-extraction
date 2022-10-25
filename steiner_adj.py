import argparse
import json
import multiprocessing
import os
import pickle

import jsonlines
import networkx as nx
from tqdm import tqdm

from conceptnet_utils import init_cpnet, concept_to_id, cpnet_simple_has_node, get_cpnet_simple, edges_between
from steiner_approximation import steiner_tree
from utils import details2adj, name_to_split


def handle_example(data):
    qc, ac, G = data
    steiner_edges = list(steiner_tree(G, qc | ac, algorithm='wu').edges)
    edges_with_rel = set()
    for n1, n2 in steiner_edges:
        edges_with_rel |= set(edges_between(n1, n2))
    return details2adj(qc, ac, edges=edges_with_rel)


def handle_file(args, fname, question_graphs):
    with jsonlines.open(fname) as f:
        qcac_for_qs = []
        for i, obj in tqdm(list(enumerate(f)), desc="loading file"):
            if isinstance(question_graphs, list):
                G = question_graphs[i]
            else:
                G = question_graphs
            qc = set(concept_to_id(c) for c in obj['qc'])
            qc = set(c for c in qc if G.has_node(c))
            ac = set(concept_to_id(c) for c in obj['ac'])
            ac = set(c for c in ac if G.has_node(c))
            qcac_for_qs.append((qc, ac, G))

    if args.n_cores > 1:
        with multiprocessing.Pool(args.n_cores) as p:
            adjs = list(tqdm(p.imap(handle_example, qcac_for_qs),
                             desc="handling questions", total=len(qcac_for_qs), smoothing=0))
    else:
        adjs = []
        for data in tqdm(qcac_for_qs, desc="handling questions"):
            adjs.append(handle_example(data))

    return adjs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-files", nargs="+")
    parser.add_argument("--cpnet-dir")
    parser.add_argument("--weight-edges", )
    parser.add_argument("--edge-id-to-edge", help="if using weighted edges, must provide the edge mapping used")
    parser.add_argument("--n-cores", default=1, type=int)
    args = parser.parse_args()

    assert False, 'superceded by result_to_adj'

    use_entire_cpnet = args.weighted_edges is None
    init_cpnet(args.cpnet_dir, simple=use_entire_cpnet, simple_biggest_component=use_entire_cpnet is None)

    # if not use_entire_cpnet:
    #     assert len(args.grounded_files) == 1, 'because you can provide multiple edge weight files for the same set of' \
    #                                           'questions, you can only provide one set of questions'
    #     with open(args.edge_id_to_edge) as f:
    #         edge_id_to_edge = json.load(f)
    #     question_graphs = []
    #     for fname in args.weighted_edges:
    #         with open(fname, 'rb') as f:
    #             all_q_data = pickle.load(f)
    #         for q_data in tqdm(all_q_data, desc="Loading per-question edge weight data"):
    #             G = nx.Graph()
    #             edge_ids, scores = q_data
    #             for edge_id, score in zip(edge_ids, scores):
    #                 s, _, o = edge_id_to_edge[edge_id]
    #                 if G.has_edge(s, o):
    #                     G[s][o]['weight'] = min(G[s][o]['weight'], float(score))
    #                 else:
    #                     G.add_edge(s, o, weight=float(score))
    #             question_graphs.append(G)
    # else:
    # question_graphs = get_cpnet_simple()

    for f in args.grounded_files:
        split = name_to_split(f)
        output_dir = "/".join(f.split("/")[:-1])

        if use_entire_cpnet:
            steiner_type = 'steiner_raw'
        else:
            steiner_type = 'steiner_weighted'

        output_f = f"{output_dir}/{steiner_type}.{split}.graph.adj.pk"
        if os.path.exists(output_f):
            continue

        adjs = handle_file(args, f)

        with open(output_f, 'wb') as f:
            pickle.dump(adjs, f)


if __name__ == '__main__':
    main()
