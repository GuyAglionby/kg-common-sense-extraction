import json
import os

import argparse
import pickle

import jsonlines
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from conceptnet_utils import fact_obj_to_strings, init_cpnet, all_edges
from embed_based_entity_linking import emb_from_words
from model.models import TripletModel
from sentence_transformers import SentenceTransformer


def do(args, model):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    edges = set()
    if args.all_concepts:
        edges = set(tuple(sorted(a)) for a in all_edges())
    else:
        print(f"Loading concepts from {args.concept_files}")
        for fname in args.concept_files:
            with jsonlines.open(fname) as f:
                for obj in tqdm(f, desc=fname):
                    edges |= {tuple(sorted(a)) for a in obj}

    print(f"{len(edges)} to embed")

    embed_file = f'{args.model_name}_embedding_embs.pt'
    embed_sro_file = f'{args.model_name}_embedding_sro.json'

    if os.path.exists(embed_sro_file):
        print("found previous embeds")
        with open(embed_sro_file, 'r') as f:
            prev_sro = json.load(f)
        prev_embs = torch.load(embed_file)
        prev_so = {tuple(sorted([e[0], e[2]])) for e in prev_sro}
        edges -= prev_so
        print(f"{len(edges)} new edges to find after subtracting existing ones")
        if len(edges) == 0:
            print("Finishing")
            return
    else:
        prev_sro = []
        prev_embs = None

    sros = []
    fact_embs = []
    edges = list(edges)

    for i in tqdm(list(range(0, len(edges), args.fact_bs)), desc="batches"):
        fact_strings, fact_sros = fact_obj_to_strings(edges[i:i + args.fact_bs])
        sros.extend(fact_sros)

        if args.sentence_transformer_model is None:
            inputs = tokenizer(
                fact_strings,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)

            if args.untrained_transformer_name is None:
                anchor_out = model.BERTModel(**inputs).detach().cpu()
            else:
                anchor_out = emb_from_words(model, inputs).detach().cpu()
        else:
            anchor_out = model.encode(fact_strings)
        fact_embs.append(anchor_out)

    if args.sentence_transformer_model is not None:
        fact_embs = torch.tensor(np.concatenate(fact_embs, axis=0))
    else:
        fact_embs = torch.cat(fact_embs, dim=0)

    if prev_embs is not None:
        fact_embs = torch.cat([prev_embs, fact_embs], dim=0)
        sros = prev_sro + sros

    torch.save(fact_embs, embed_file)
    with open(embed_sro_file, 'w') as f:
        json.dump(sros, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-files", nargs='+')
    parser.add_argument("--cpnet_folder", type=str)
    parser.add_argument("--fact-bs", default=64, type=int)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--all-concepts", action='store_true')
    parser.add_argument("--sentence-transformer-model", type=str)
    parser.add_argument("--untrained-transformer-name", type=str)

    args = parser.parse_args()
    init_cpnet(args.cpnet_folder)
    model_path = 'save_model/recall'
    for path in os.listdir(model_path):
        if path != 'roberta':
            continue
        args_path = os.path.join(model_path, path, 'args.json')
        params = json.load(open(args_path, 'r'))
        args.output_dir = params['output_dir']
        args.bert_path = params['bert_path']
        args.device = torch.device("cuda")

        if args.sentence_transformer_model is None:
            if args.untrained_transformer_name is None:
                model = TripletModel(
                    bert_model=args.bert_path,
                )
                state_dict = torch.load(os.path.join(args.output_dir, 'model_best.bin'))
                model.load_state_dict(state_dict)
            else:
                model = AutoModel.from_pretrained(args.untrained_transformer_name)
        else:
            model = SentenceTransformer(args.sentence_transformer_model)
        model.cuda()
        with torch.no_grad():
            model.eval()
            do(args, model)


if __name__ == '__main__':
    main()
