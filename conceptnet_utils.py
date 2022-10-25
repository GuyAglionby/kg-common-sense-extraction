import glob

import networkx as nx

merged_relations = [
    'antonym', 'atlocation', 'capableof', 'causes', 'createdby', 'isa', 'desires', 'hassubevent', 'partof',
    'hascontext', 'hasproperty', 'madeof', 'notcapableof', 'notdesires', 'receivesaction', 'relatedto', 'usedfor'
]
concept2id, id2concept, relation2id, id2relation, cpnet, cpnet_simple = None, None, None, None, None, None


def get_cpnet_simple(copy=False):
    assert cpnet_simple is not None
    if copy:
        return cpnet_simple.copy()
    else:
        return cpnet_simple


def init_cpnet(cpnet_folder, simple=False, simple_biggest_component=False):
    if concept2id is None:
        load_resources(f"{cpnet_folder}/entity_vocab.txt", 'cpnet')
    if cpnet is None or (cpnet_simple is None and simple):
        graph_file = list(glob.glob(f"{cpnet_folder}/*.en.pruned.graph"))[0]
        load_cpnet(graph_file, simple, simple_biggest_component)


def load_cpnet(cpnet_graph_path, simple=False, simple_biggest_component=False):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    if simple:
        cpnet_simple = nx.Graph()
        cpnet_edges = list(cpnet.edges())
        for u, v in cpnet_edges:
            cpnet_simple.add_edge(u, v)
        if simple_biggest_component:
            components = list(nx.connected_components(cpnet_simple))
            components.sort(key=len)
            cpnet_simple = cpnet_simple.subgraph(components[-1])


def load_resources(cpnet_vocab_path, graph):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    for c in id2concept:
        if ' ' in c:
            concept2id[c.replace(' ', '_')] = concept2id[c]
        if '_' in c:
            concept2id[c.replace('_', ' ')] = concept2id[c]

    if graph == 'cpnet':
        id2relation = merged_relations
    else:
        # relations.tsv from
        relation_vocab_path = cpnet_vocab_path.replace("entity_vocab.txt", "relations.tsv")
        with open(relation_vocab_path) as f:
            id2relation = [x.split("\t")[1].strip() for x in f]
    relation2id = {r: i for i, r in enumerate(id2relation)}


def cpnet_simple_has_node(u):
    return cpnet_simple.has_node(u)


def concept_to_id(concept):
    return concept2id[concept]


def convert_triple_to_sentence_cpnet(s, r, o):
    if r == 'antonym':
        return f'{s} and {o} are opposites'
    elif r == 'atlocation':
        return f'{s} is a typical location for {o}'
    elif r == 'capableof':
        return f'Something that {s} can typically do is {o}'
    elif r == 'causes':
        return f'{s} and {o} are events, and it is typical for {s} to cause {o}'
    elif r == 'createdby':
        return f'{o} is a process or agent that creates {s}'
    elif r == 'isa':
        return f'{s} is a subtype or a specific instance of {o}'
    elif r == 'desires':
        return f'{s} is a conscious entity that typically wants {o}'
    elif r == 'hassubevent':
        return f'{s} and {o} are events, and {o} happens as a subevent of {s}'
    elif r == 'partof':
        return f'{s} is a part of {o}'
    elif r == 'hascontext':
        return f'{s} is a word used in the context of {o}, which could be a topic area, technical field, or regional dialect'
    elif r == 'hasproperty':
        return f'{s} has {o} as a property'
    elif r == 'madeof':
        return f'{s} is made of {o}'
    elif r == 'notcapableof':
        return f'Something that {s} cannot typically do is {o}'
    elif r == 'notdesires':
        return f'{s} is a conscious entity that does not typically want {o}'
    elif r == 'receivesaction':
        return f'{o} can be done to {s}'
    elif r == 'relatedto':
        return f'There is some positive relationship between {s} and {o}'
    elif r == 'usedfor':
        return f'{s} is used for {o}'
    else:
        raise ValueError(r)


def num_relations():
    return len(relation2id)


def cpnet_has_edge(s, t):
    return cpnet.has_edge(s, t)


def all_edges():
    return cpnet.edges()


def edges_between(s, t):
    edges = []
    for r in rels_between(s, t):
        edges.append((s, r, t))
    for r in rels_between(t, s):
        edges.append((t, r, s))
    return edges


def rels_between(s, t):
    rels = []
    edge_data = cpnet.get_edge_data(s, t)
    if edge_data is None:
        return rels
    for e_attr in edge_data.values():
        if e_attr['rel'] >= 0 and e_attr['rel'] < num_relations():
            rels.append(e_attr['rel'])
    return rels


def fact_obj_to_strings(obj):
    sents = []
    sro = []
    for s, o in obj:
        ss = id2concept[s].replace("_", " ")
        oo = id2concept[o].replace("_", " ")
        added_rels = set()
        for r in [v['rel'] for v in cpnet.get_edge_data(s, o).values()]:
            if r < 17:
                if r in added_rels:
                    continue
                sents.append(convert_triple_to_sentence_cpnet(ss, id2relation[r], oo))
                sro.append((s, r, o))
                added_rels.add(r)
            else:
                if r - 17 in added_rels:
                    continue
                sents.append(convert_triple_to_sentence_cpnet(oo, id2relation[r - 17], ss))
                sro.append((o, r - 17, s))
                added_rels.add(r - 17)

    return sents, sro


