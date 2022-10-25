import argparse
import pickle
from collections import Counter

import networkx as nx
import numpy as np
from networkx import NetworkXNoPath


def load_adj_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    out = []
    for question in data:
        assert isinstance(question, dict)
        rel_adj = question['adj']
        nodes = question['concepts']
        n_nodes = rel_adj.shape[1]
        rel_adj_flat = rel_adj.toarray().reshape((-1, n_nodes, n_nodes)).nonzero()
        relations = rel_adj_flat[0].tolist()
        sources = nodes[rel_adj_flat[1]].tolist()
        targets = nodes[rel_adj_flat[2]].tolist()
        qc = set(question['concepts'][question['qmask']].tolist())
        ac = set(question['concepts'][question['amask']].tolist())
        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        for s, _, o in set(zip(sources, relations, targets)):
            G.add_edge(s, o)
        out.append((G, qc, ac))
    return out


def any_qcac_path(G, qc, ac):
    no_path = 0
    path = 0
    for q in qc:
        for a in ac:
            gen = nx.shortest_simple_paths(G, q, a)
            try:
                next(gen)
                path += 1
            except NetworkXNoPath:
                no_path += 1
    return path, no_path


def qcac_degree(G, qc, ac):
    qc_degrees = [len(G.edges(q)) for q in qc]
    ac_degrees = [len(G.edges(a)) for a in ac]

    qc_any_nonzero = any(a > 0 for a in qc_degrees)
    ac_any_nonzero = any(a > 0 for a in ac_degrees)

    return np.mean(qc_degrees), np.mean(ac_degrees) if len(ac_degrees) else 0, qc_any_nonzero, ac_any_nonzero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj')
    args = parser.parse_args()

    graphs = load_adj_pickle(args.adj)

    proportion_pairs_have_path = []
    any_pair_has_path = 0
    qc_deg, ac_deg = [], []
    qc_nonzero_deg, ac_nonzero_deg = [], []
    avg_degree = []

    for G, qc, ac in graphs:
        has_path, not_has_path = any_qcac_path(G, qc, ac)
        if has_path > 0:
            any_pair_has_path += 1
        if has_path + not_has_path > 0:
            proportion_pairs_have_path.append(has_path / (has_path + not_has_path))

        mean_qc_deg, mean_ac_deg, any_qc_nonzero, any_ac_nonzero = qcac_degree(G, qc, ac)
        qc_deg.append(mean_qc_deg)
        qc_nonzero_deg.append(any_qc_nonzero)
        ac_deg.append(mean_ac_deg)
        ac_nonzero_deg.append(any_ac_nonzero)
        avg_degree.append(np.mean([G.degree[n] for n in G.nodes()]))

    print(f"Of {len(graphs)} questions+answers, {any_pair_has_path} have a path between at least 1 QC an AC ({any_pair_has_path/len(graphs)})")
    print(f"Macro average proportion of QC/AC pairs that are connected: {np.mean(proportion_pairs_have_path)}")
    print()
    print(f"Macro average degree: {np.mean(avg_degree)} ({np.std(avg_degree)})")
    print()
    print(f"Macro average qc degree: {np.mean(qc_deg)}")
    print(f"Macro average ac degree: {np.mean(ac_deg)}")
    print()
    print(f"Questions with >=1 nonzero-degree QC: {Counter(qc_nonzero_deg)}")
    print(f"Questions with >=1 nonzero-degree AC: {Counter(ac_nonzero_deg)}")


if __name__ == '__main__':
    main()
