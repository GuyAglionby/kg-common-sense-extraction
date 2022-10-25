import numpy as np
import torch
from scipy.sparse import coo_matrix

from conceptnet_utils import num_relations, cpnet_has_edge, rels_between


def name_to_split(name):
    if 'dev' in name:
        split = 'dev'
    elif 'train' in name:
        split = 'train'
    elif 'test' in name:
        split = 'test'
    else:
        raise ValueError(name)
    return split


def emb_from_words(model, tokenized_input):
    outs = model(**tokenized_input).last_hidden_state[:, 1:-1].detach().cpu()
    attention = tokenized_input['attention_mask'][:, 1:-1]
    returned_items = []
    for emb, atn in zip(outs, attention):
        returned_items.append(emb[atn.bool()].mean(0).detach().cpu())
    return torch.stack(returned_items)


def concepts2adj(node_ids, edges=None):
    """
    takes a list of concept IDs
    for all pairs, looks to see if there's an edge between them

    adapted from MHGRN/utils/graph.py
    """
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = num_relations()
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    if edges is None:
        for s in range(n_node):
            for t in range(n_node):
                s_c, t_c = cids[s], cids[t]
                if cpnet_has_edge(s_c, t_c):
                    for e in rels_between(s_c, t_c):
                        adj[e][s][t] = 1
    else:
        idxs = {x: np.where(cids == x)[0].squeeze().item() for x in node_ids}
        for s, r, o in edges:
            assert s in node_ids
            assert o in node_ids
            adj[r][idxs[s]][idxs[o]] = 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj


def details2adj(qc, ac, edges=None, nodes=None):
    assert edges is not None or nodes is not None
    if edges is not None:
        all_concepts = set()
        for s, _, o in edges:
            all_concepts |= {s, o}
    else:
        all_concepts = set(nodes)

    qc = set(qc)
    ac = set(ac)

    concepts = list(qc) + list(ac)
    all_concepts -= qc
    all_concepts -= ac
    concepts += list(all_concepts)
    concepts = np.asarray(concepts)

    if edges is not None:
        adj = concepts2adj(concepts, edges=edges)
    else:
        adj = concepts2adj(concepts)

    return {
        'concepts': concepts,
        'qmask': np.asarray([c in qc for c in concepts]),
        'amask': np.asarray([c in ac for c in concepts]),
        'adj': adj
    }
