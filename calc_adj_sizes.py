import argparse
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Calculate how many nodes/edges are in different subgraphs")
    parser.add_argument("--dirs", nargs="+", help="Directories containing MHGRN-style .adj.pk files")
    args = parser.parse_args()

    files = ['train.graph.adj.pk', 'dev.graph.adj.pk', 'test.graph.adj.pk']
    results = []

    for d in tqdm(args.dirs, desc='Schema graph types'):
        n_nodes, n_edges = [], []
        for f in files:
            filename = f"{d}/{f}"
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
                for q in obj:
                    n_nodes.append(len(q['concepts']))
                    n_edges.append(q['adj'].nnz)
                    break
        results.append({
            'filename': d,
            'n': len(n_nodes),
            'n_nodes_mean': np.mean(n_nodes),
            'n_nodes_std': np.std(n_nodes),
            'n_nodes_median': np.median(n_nodes),
            'n_nodes_lq': np.quantile(n_nodes, 0.25),
            'n_nodes_uq': np.quantile(n_nodes, 0.75),
            'n_edges_mean': np.mean(n_edges),
            'n_edges_std': np.std(n_edges),
            'n_edges_median': np.median(n_edges),
            'n_edges_lq': np.quantile(n_edges, 0.25),
            'n_edges_uq': np.quantile(n_edges, 0.75),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('schema_graph_types_statistics.csv', index=False)


if __name__ == '__main__':
    main()
