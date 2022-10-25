import argparse
import glob
import itertools
import json
import multiprocessing
import os
import pickle
from collections import defaultdict
from functools import partial
from heapq import heappush, heappop

import jsonlines
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from conceptnet_utils import concept_to_id, init_cpnet, get_cpnet_simple, edges_between, cpnet_simple_has_node
from steiner_approximation import steiner_tree, HeapEntry
from utils import name_to_split, details2adj


global per_q_fact_scores, fact_order


def get_next_path_weight(G, gen):
    try:
        path = next(gen)
        gen = itertools.chain([path], gen)
        w = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
        return gen, w
    except StopIteration:
        return gen, -1
    except nx.NetworkXNoPath:
        return gen, -1


def handle_question(args, data):
    edges, qcac, question_idx = data
    concepts = []

    if args.steiner or args.pathfinding:
        # we use only the largest connected component when finding the steiner spanning tree
        qcac['qc'] = [c for c in qcac['qc'] if cpnet_simple_has_node(c)]
        qcac['ac'] = [c for c in qcac['ac'] if cpnet_simple_has_node(c)]

    concepts.extend(qcac['qc'])
    concepts.extend(qcac['ac'])
    edge_concepts = set()
    concepts_union = set(concepts)
    keep_edges = set()
    for e in edges:
        if args.steiner and (not cpnet_simple_has_node(e[0]) or not cpnet_simple_has_node(e[2])):
            continue
        if len(concepts_union | {e[0], e[2]}) <= args.max_nodes:
            keep_edges.add(e)
            concepts_union |= {e[0], e[2]}
            edge_concepts.add(e[0])
            edge_concepts.add(e[2])
    edge_concepts -= set(concepts)
    concepts.extend(list(edge_concepts))

    stats = {}

    if args.pathfinding:
        assert len(edges) == 0, "weird to use pathfinding when you already have edges"
        G = get_cpnet_simple(copy=True)
        if args.global_weights_idx is not None:
            fact_scores_for_q = per_q_fact_scores[question_idx]
            for (s, _, o), weight in zip(fact_order, fact_scores_for_q):
                if G.has_edge(s, o):
                    G[s][o]['weight'] = min(weight, G[s][o].get('weight', float('inf')))

        gens = []
        for q in qcac['qc']:
            for a in qcac['ac']:
                gen = nx.shortest_simple_paths(G, q, a, weight='weight')
                gen, weight = get_next_path_weight(G, gen)
                if weight >= 0:
                    heappush(gens, HeapEntry(weight, gen))

        added_nodes = set()
        while len(gens) and len(added_nodes) < args.max_nodes:
            gen = heappop(gens)
            gen = gen.data
            path = next(gen)
            # # hardcoded max path length of 4 (= 5 nodes)
            # if len(path) > 5:
            #     continue
            added_nodes |= set(path)
            next_gen, next_weight = get_next_path_weight(G, gen)
            if next_weight >= 0:
                heappush(gens, HeapEntry(next_weight, next_gen))
        return details2adj(qcac['qc'], qcac['ac'], nodes=added_nodes), stats

    if args.steiner:
        existing_G = nx.Graph()
        for s, _, o in keep_edges:
            existing_G.add_edge(s, o)
        G = get_cpnet_simple(copy=True)

        # Preserve the edges we have already
        contracted_node_mapping = defaultdict(set)
        contracted_nodes = set()
        for component in nx.connected_components(existing_G):
            nodes = list(component)
            for to_collapse in nodes[1:]:
                contracted_node_mapping[nodes[0]].add(to_collapse)
                contracted_nodes.add(to_collapse)
                nx.contracted_nodes(G, nodes[0], to_collapse,
                                    copy=False, self_loops=False)

        if args.global_weights_idx is not None:
            fact_scores_for_q = per_q_fact_scores[question_idx]
            for (s, _, o), weight in zip(fact_order, fact_scores_for_q):
                if G.has_edge(s, o):
                    G[s][o]['weight'] = min(weight, G[s][o].get('weight', float('inf')))

        # Find the minimum(ish) spanning graph
        terminals = [c for c in concepts if c not in contracted_nodes]
        w = steiner_tree(G, terminals, algorithm="wu")
        steiner_graph = list(w.edges)

        # undo our contraction
        steiner_addtl_edges = set()
        steiner_nodes = set()
        for n1, n2 in steiner_graph:
            n1_nodes = {n1} | contracted_node_mapping.get(n1, set())
            n2_nodes = {n2} | contracted_node_mapping.get(n1, set())
            for new_n1, new_n2 in itertools.product(n1_nodes, n2_nodes):
                steiner_addtl_edges |= set(edges_between(new_n1, new_n2))
                steiner_nodes |= {new_n1, new_n2}

        steiner_nodes -= set(concepts)
        concepts.extend(list(steiner_nodes))
        steiner_addtl_edges -= keep_edges
        keep_edges |= steiner_addtl_edges
        stats['steiner_added_edges'] = len(steiner_addtl_edges)
        stats['steiner_added_nodes'] = len(steiner_nodes)

    stats['num_nodes'] = len(concepts)
    stats['num_edges'] = len(keep_edges)

    return details2adj(qcac['qc'], qcac['ac'], edges=keep_edges), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="expected to be `result/xyz`")
    parser.add_argument("--edge_embed_method", help="not required if dir is provided")
    parser.add_argument("--edge-emb-id-mapping-dir", default='.')
    parser.add_argument("--grounded-dir")
    parser.add_argument("--cpnet-dir")
    parser.add_argument("--addtl-save-string", default='')
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs='+', default=['train', 'test', 'dev'])
    parser.add_argument("--steiner", action="store_true")
    parser.add_argument("--pathfinding", action="store_true")
    parser.add_argument("--global-weights-idx", type=int, default=None)
    parser.add_argument("--edge-score-percentile", default=0, type=float)
    parser.add_argument("--max-edges", default=200, type=int)
    parser.add_argument("--max-nodes", default=200, type=int, help="this is overloaded - both for loading from `dir` and also for pathfinding")
    parser.add_argument("--n-cores", default=1, type=int)
    parser.add_argument("--save-chunksizes", default=None, type=int,
                        help="when doing steiner/pathfinding, save intermediate results in chunks")
    parser.add_argument("--start-chunk", default=None, type=int,)
    parser.add_argument("--end-chunk", default=None, type=int, )
    parser.add_argument("--global_weights_bs", type=int)
    parser.add_argument("--global_weights_idx-use-raw", action='store_true')
    args = parser.parse_args()

    assert not (args.steiner and args.pathfinding), 'cant do both!'
    assert not (args.save_chunksizes is not None and args.global_weights_bs is not None), 'save chunksizes only when ' \
                                                                                          'not already chunking ' \
                                                                                          'because of global weights'

    assert not (args.dir is None and not args.steiner and not args.pathfinding), 'if no dir provided must be ' \
                                                                                 'pathfinding or steiner'

    init_cpnet(args.cpnet_dir, simple=args.steiner or args.pathfinding, simple_biggest_component=args.steiner or args.pathfinding)

    if args.dir is not None:
        if args.dir.endswith("/"):
            args.dir = args.dir[-1]
        method_details = args.dir.split("/")[-1]
        edge_embed_method = method_details.split("_")[0]
        if args.edge_embed_method is not None and args.edge_embed_method != edge_embed_method:
            print(f'providing an embed method of {args.edge_embed_method} is a NO-OP; using method as specified with '
                  f'dir `{edge_embed_method}`')
    else:
        assert args.edge_embed_method is not None, 'if not providing dir must provide edge embed method'
        edge_embed_method = args.edge_embed_method
        method_details = f"{edge_embed_method}_no-dir"

    if args.steiner:
        steiner_str = '-steiner'
    elif args.pathfinding:
        steiner_str = '-pathfinding'
    else:
        steiner_str = ''
    if args.global_weights_idx is not None:
        assert args.global_weights_idx >= 0
        assert args.global_weights_bs is not None, "must provide batch size"
        steiner_str += f'-global-weights'

    output_dir = f"graph_adjs/{args.dataset}-{method_details}-{args.max_edges}edges-{args.max_nodes}nodes-{args.edge_score_percentile}edge_percentile{steiner_str}{args.addtl_save_string}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{args.edge_emb_id_mapping_dir}/{edge_embed_method}_embedding_sro.json") as f:
        edge_id_mapping = [tuple(sro) for sro in json.load(f)]

    # Load edges + their scores from the results directory (output of sort_predict.py)
    if args.dir is not None:
        if not os.path.exists(args.dir):
            raise ValueError(f"{args.dir} does not exist")
        split_to_edges = defaultdict(list)
        all_scores = []
        for filename in glob.glob(f"{args.dir}/*pkl"):
            split = name_to_split(filename.split("/")[-1])

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            for edges, scores in data:
                edges = [(edge_id_mapping[e], s) for e, s in zip(edges[:args.max_edges], scores[:args.max_edges])]
                if split == 'train':
                    all_scores.append(scores)
                split_to_edges[split].append(edges)

        # Apply a percentile-based threshold (from train set) to edge scores
        if len(all_scores):
            score_threshold = np.quantile(np.concatenate(all_scores), args.edge_score_percentile)
            for split, questions in split_to_edges.items():
                split_to_edges[split] = [[edge for edge, score in question if score >= score_threshold]
                                         for question in questions]
        elif args.edge_score_percentile != 0:
            raise ValueError("train edges not loaded so can't apply edge score percentile")

    # Load qc/ac from grounded files
    split_to_qcac = defaultdict(list)
    for filename in glob.glob(f"{args.grounded_dir}/*.grounded.jsonl"):
        split = name_to_split(filename.split("/")[-1])
        with jsonlines.open(filename) as f:
            for obj in f:
                qc = [concept_to_id(concept) for concept in obj['qc']]
                ac = [concept_to_id(concept) for concept in obj['ac']]
                qc = [c for c in qc if c not in ac]
                split_to_qcac[split].append({'qc': qc, 'ac': ac})

    # Load in question + fact embeddings
    if args.global_weights_idx is not None:
        assert len(args.splits) == 1, 'when using global weights can only use one split'
        steiner_q_start_idx = args.global_weights_idx * args.global_weights_bs
        print(f"Loading question-fact scores for index {args.global_weights_idx} "
              f"(Q{steiner_q_start_idx}-{steiner_q_start_idx + args.global_weights_bs})")
        global per_q_fact_scores, fact_order

        file_idx = args.global_weights_idx if args.global_weights_idx_use_raw else steiner_q_start_idx
        per_q_fact_scores = torch.load(f"per_q_edge_scores_{edge_embed_method}_{args.dataset}/{args.splits[0]}_{file_idx}.pt")

        with open(f"{edge_embed_method}_embedding_sro.json") as f:
            fact_order = json.load(f)
    else:
        steiner_q_start_idx = None

    # actually do something now we've loaded
    for split in args.splits:
        if steiner_q_start_idx is not None:
            split_with_chunk_idx = f"{split}-{steiner_q_start_idx}"
        else:
            split_with_chunk_idx = f"{split}"
        output_adj_file = f'{output_dir}/{split_with_chunk_idx}.graph.adj.pk'
        if os.path.exists(output_adj_file):
            continue

        split_qcac = split_to_qcac[split]
        if steiner_q_start_idx is not None:
            split_qcac = split_qcac[steiner_q_start_idx:steiner_q_start_idx + args.global_weights_bs]

        if args.dir is not None:
            split_question_edges = split_to_edges[split]
            if steiner_q_start_idx is not None:
                split_question_edges = split_question_edges[steiner_q_start_idx:steiner_q_start_idx + args.global_weights_bs]
        else:
            split_question_edges = [[]] * len(split_qcac)
        data_zip = list(zip(split_question_edges, split_qcac, list(range(len(split_question_edges)))))
        all_stats = defaultdict(list)

        if args.n_cores > 1:
            handle_question_partial = partial(handle_question, args)

            data_zip_iterator_start = 0
            if args.save_chunksizes is None:
                data_zip_chunks = [data_zip]
            else:
                data_zip_chunks = [data_zip[k:k + args.save_chunksizes] for k in range(0, len(data_zip), args.save_chunksizes)]
                assert (args.start_chunk is None and args.end_chunk is None) or (args.start_chunk is not None and args.end_chunk is not None)
                if args.start_chunk is not None:
                    data_zip_chunks = data_zip_chunks[args.start_chunk:args.end_chunk]
                    data_zip_iterator_start = args.start_chunk

            n_chunks = len(data_zip_chunks)

            outer_q_datas = None
            for j, chunk in enumerate(data_zip_chunks, data_zip_iterator_start):
                with multiprocessing.Pool(args.n_cores) as p:
                    results = list(tqdm(p.imap(handle_question_partial, chunk),
                                        desc="question processing", total=len(split_question_edges), smoothing=0))

                inner_q_datas, q_stats = zip(*results)
                inner_q_datas = list(inner_q_datas)
                for stats_inst in q_stats:
                    for k, v in stats_inst.items():
                        all_stats[k].append(v)

                if n_chunks > 1:
                    with open(output_adj_file + f"{j}", 'wb') as f:
                        pickle.dump(inner_q_datas, f)
                else:
                    outer_q_datas = inner_q_datas
        else:
            assert args.save_chunksizes is None, 'not implemented'
            outer_q_datas = []
            for q_data in tqdm(data_zip, desc="question processing", total=len(split_question_edges)):
                q_adj, q_stats = handle_question(args, q_data)
                for k, v in q_stats.items():
                    all_stats[k].append(v)
                outer_q_datas.append(q_adj)

        if outer_q_datas is not None:
            with open(output_adj_file, 'wb') as f:
                pickle.dump(outer_q_datas, f)

        final_stats = {f'avg_{k}': np.mean(v) for k, v in all_stats.items()}
        if args.steiner:
            final_stats['steiner_node_proportion'] = [steiner_n / total_n for steiner_n, total_n
                                                      in zip(all_stats['steiner_added_nodes'], all_stats['num_nodes'])]
            final_stats['steiner_edge_proportion'] = [steiner_n / total_n for steiner_n, total_n
                                                      in zip(all_stats['steiner_added_edges'], all_stats['num_edges'])]
        with open(f'{output_dir}/{split_with_chunk_idx}-stats.json', 'w') as f:
            json.dump(final_stats, f)


if __name__ == '__main__':
    main()
