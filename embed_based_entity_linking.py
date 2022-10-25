import argparse
import os
from collections import defaultdict

import jsonlines
import numpy as np
import spacy
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import name_to_split, emb_from_words

nlp = spacy.load('en_core_web_sm')


def get_vp(doc, tok_idx):
    toks = [doc[tok_idx]]
    tok_queue = [doc[tok_idx]]
    acceptable_deps = {'aux', 'dobj', 'nsubj', 'det'}
    while len(tok_queue):
        tok = tok_queue.pop()
        for child in tok.children:
            if child.dep_.lower() in acceptable_deps:
                toks.append(child)
                tok_queue.append(child)
    tok_idxs = [t.i for t in toks]
    max_t = max(tok_idxs)
    min_t = min(tok_idxs)
    return doc[min_t:max_t + 1].text


def get_vps(args, doc):
    vps = []
    for tok in doc:
        if tok.pos_ == 'VERB' and tok.dep_ != 'aux':
            if args.include_verb_phrases:
                vps.append(get_vp(doc, tok.i))
            vps.append(tok.text)
    return vps


def synonyms_for_line(args, line):
    syns = [line]
    if args.include_noun_phrases or args.include_verb_phrases or args.include_verbs:
        doc = nlp(line)
    if args.include_noun_phrases:
        for np in doc.noun_chunks:
            if '-PRON-' in np.lemma_:
                continue
            syns.append(np.text)
    if args.include_verb_phrases or args.include_verbs:
        syns.extend(get_vps(args, doc))
    return syns


def reconcile_predictions(entity_list, score_list):
    entity_to_score = {}
    for e, s in zip(entity_list, score_list):
        entity_to_score[e] = min(entity_to_score.get(e, s + 1), s)

    entity_with_score = list(entity_to_score.items())
    entity_with_score.sort(key=lambda x: x[1])
    return entity_with_score[:1000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statements", nargs="+")
    parser.add_argument("--cpnet-dir")
    parser.add_argument("--model")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default='./emb_grounding_results/')
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--max-entities-per-phrase", default=5, type=int)
    parser.add_argument("--include-verbs", action="store_true")
    parser.add_argument("--include-verb-phrases", action="store_true")
    parser.add_argument("--include-noun-phrases", action="store_true")
    args = parser.parse_args()

    this_emb_output_dir = f'{args.dataset}_emb_grounding'
    if args.include_noun_phrases:
        this_emb_output_dir += '-nps'
    if args.include_verb_phrases:
        this_emb_output_dir += '-vps'
    if args.include_verbs:
        this_emb_output_dir += '-verbs'

    args.output_dir = args.output_dir + "/" + this_emb_output_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(f"{args.cpnet_dir}/entity_vocab.txt") as f:
        vocab = [a for a in f.read().split("\n") if len(a)]

    model = AutoModel.from_pretrained(args.model).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    emb_mat_file = f'embed-entity-linking-matrix_{args.model}.pt'

    if not os.path.exists(emb_mat_file):
        bs = 32
        emb_mat = []
        for i in tqdm(list(range(0, len(vocab), bs))):
            batch = {k: v.cuda() for k, v in tokenizer(vocab[i:i + bs], return_tensors='pt', padding=True).items()}
            # cls doens't work well; pool over word embeddings instead
            emb_mat.append(emb_from_words(model, batch))

        emb_mat = torch.cat(emb_mat)
        torch.save(emb_mat, emb_mat_file)
    else:
        emb_mat = torch.load(emb_mat_file)

    with torch.no_grad():
        model.eval()
        for s in tqdm(args.statements, desc="Files"):
            split = name_to_split(s)
            output_file = f"{args.output_dir}/{split}_emb_grounded.jsonl"
            if os.path.exists(output_file):
                continue
            questions = []
            answers = []
            with jsonlines.open(s) as f:
                for i, obj in tqdm(list(enumerate(f)), desc="Lines in statement file"):
                    questions.extend([(i, s) for s in synonyms_for_line(args, obj['question']['stem'])])
                    for j, ans in enumerate(obj['question']['choices']):
                        answers.extend([((i, j), s) for s in synonyms_for_line(args, ans['text'])])

            emb_outputs = defaultdict(lambda: defaultdict(list))
            to_embed = questions + answers
            for i in tqdm(list(range(0, len(to_embed), args.bs)), desc="Embedding"):
                target_idxs, targets = zip(*to_embed[i:i+args.bs])
                batch = tokenizer(list(targets),
                                  padding=True,
                                  return_tensors='pt')
                for k, v in batch.items():
                    batch[k] = v.cuda()

                output = emb_from_words(model, batch)

                entity_distances = torch.cdist(output, emb_mat)
                closest_entities = entity_distances.argsort(dim=1)[:, :args.max_entities_per_phrase]
                closest_entity_scores = entity_distances.gather(1, closest_entities).tolist()
                closest_entities = closest_entities.tolist()

                for targ_idx, ent_list, scores in zip(target_idxs, closest_entities, closest_entity_scores):
                    emb_outputs[targ_idx]['ent_list'].extend([vocab[w] for w in ent_list])
                    emb_outputs[targ_idx]['scores'].extend(scores)

            # we should be able to use a list for the answers, as the order is sorted coming through the model,
            # but i'm rushing and don't want to make a mistake.
            cleaned_emb_outputs = defaultdict(lambda: {'question': [], 'answers': {}})
            for k, v in emb_outputs.items():
                reconciled = reconcile_predictions(v['ent_list'], v['scores'])
                if isinstance(k, int):
                    cleaned_emb_outputs[k]['question'] = reconciled
                else:
                    cleaned_emb_outputs[k[0]]['answers'][k[1]] = reconciled

            cleaned_emb_outputs = list(cleaned_emb_outputs.items())
            cleaned_emb_outputs.sort(key=lambda x: x[0])
            cleaned_emb_outputs = [v for k, v in cleaned_emb_outputs]
            with jsonlines.open(output_file, 'w') as f:
                f.write_all(cleaned_emb_outputs)


if __name__ == '__main__':
    main()