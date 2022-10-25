import argparse
import glob
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+")
    args = parser.parse_args()

    for d in args.dirs:
        stats = glob.glob(f"{d}/train*stats.json")
        n = 0
        avg_steiner_added_edges = 0
        avg_steiner_added_nodes = 0
        avg_num_nodes = 0
        avg_num_edges = 0
        steiner_node_proportion = []
        steiner_edge_proportion = []
        for sf in stats:
            with open(sf) as f:
                obj = json.load(f)
            steiner_node_proportion.extend(obj['steiner_node_proportion'])
            steiner_edge_proportion.extend(obj['steiner_edge_proportion'])
            here_n = len(obj['steiner_edge_proportion'])
            assert len(obj['steiner_node_proportion']) == len(obj['steiner_edge_proportion'])
            n += here_n
            avg_num_edges += obj['avg_num_edges'] * here_n
            avg_num_nodes += obj['avg_num_nodes'] * here_n
            avg_steiner_added_edges += obj['avg_steiner_added_edges'] * here_n
            avg_steiner_added_nodes += obj['avg_steiner_added_nodes'] * here_n

        avg_steiner_added_edges /= n
        avg_steiner_added_nodes /= n
        avg_num_nodes /= n
        avg_num_edges /= n
        steiner_node_proportion = np.mean(steiner_node_proportion)
        steiner_edge_proportion = np.mean(steiner_edge_proportion)

        print(d)
        print(f"{avg_steiner_added_edges=}")
        print(f"{avg_steiner_added_nodes=}")
        print(f"{avg_num_nodes=}")
        print(f"{avg_num_edges=}")
        print(f"{steiner_node_proportion=}")
        print(f"{steiner_edge_proportion=}")

if __name__ == '__main__':
    main()
