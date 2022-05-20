import argparse
import json
import os
import pickle
import random
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from dataset import *
import logging
import torch.nn.functional as F

from model.models import TripletModel
from utils import name_to_split

from evaluate import eval_ndcg, eval_ndcg_train
logger = logging.getLogger()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_qa_dataset(qa_dataset, model, tokenizer, args, split_name):
    data_loader = DataLoader(qa_dataset, batch_size=args.batch_size,
                             collate_fn=DataCollatorForTest(tokenizer, args=args, mode=split_name))
    data_loader = tqdm(data_loader)
    val_preds = []
    for inputs in tqdm(list(data_loader), desc="encoding question reps"):
        if args.sentence_transformer_model is None:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)
            anchor_out = model.BERTModel(**inputs).detach()
        else:
            anchor_out = model.encode(inputs, convert_to_tensor=True, show_progress_bar=False)
        val_preds.extend(anchor_out)
    return val_preds


def model_predict(args, qa_dataset, fact_dataset, split_name, model):
    output_file = f'{args.save_model_path}/recall_scores_{split_name}.pkl'
    if os.path.isfile(output_file):
        return

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    # Encoding the facts
    facts_preds = []
    fact_loader = DataLoader(fact_dataset, batch_size=args.batch_size,
                             collate_fn=DataCollatorForExplanation(tokenizer))
    fact_loader = tqdm(fact_loader)
    for batch_index, inputs in enumerate(fact_loader):  # 调用模型计算facts表示
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        facts_preds.append(anchor_out)
    facts_preds = torch.cat(facts_preds, dim=0)

    # Encoding the QA set
    # qa_dataset is just a list of question IDs
    val_preds = encode_qa_dataset(qa_dataset, model, tokenizer, args, split_name)

    # Nearest neighbours
    scores = []
    for index, (pred, data) in enumerate(zip(val_preds, qa_dataset)):
        score = F.pairwise_distance(pred, facts_preds, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    pd.to_pickle(scores, output_file)


def load_pretrained_fact_embs(args):
    nodes_to_edges = defaultdict(list)
    edge_to_idx = {}
    embedding_mat = torch.load(args.edge_emb_path)
    with open(args.edge_emb_mapping_path) as f:
        for edge in json.load(f):
            edge = tuple(edge)
            edge_to_idx[edge] = len(edge_to_idx)
            nodes_to_edges[tuple(sorted([edge[0], edge[2]]))].append(edge)
    return nodes_to_edges, edge_to_idx, embedding_mat


def model_predict_qa(args, qa_dataset, split_name, model):
    statement_file = args.statement_path.split("/")[-1]
    split = name_to_split(statement_file)

    pretrained_embs_name = args.edge_emb_path.split("/")[-1].replace("_embedding_embs.pt", "")

    e4s = '' if args.edges_for_statements is not None else '_all_edges'

    output_file = f'{args.save_model_path}/recall_scores_{split_name}_{split}_{pretrained_embs_name}{e4s}.pkl'
    if os.path.isfile(output_file):
        return

    print("loading pre-trained embs")
    nodes_to_edges, edge_to_idx, embedding_mat = load_pretrained_fact_embs(args)
    print("done")

    with torch.no_grad():
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

        # Encoding the QA set
        # qa_dataset is just a list of question IDs
        val_preds = encode_qa_dataset(qa_dataset, model, tokenizer, args, split_name)

        scores = []
        sros = []
        if args.edges_for_statements is not None:
            with jsonlines.open(args.edges_for_statements) as f:
                for pred, fact_obj in tqdm(list(zip(val_preds, f)), desc="per-question edge finding (from paths)"):
                    sros.append([])
                    resulting_embs = []
                    for nodes in fact_obj:
                        edges = nodes_to_edges[tuple(sorted(nodes))]
                        for e in edges:
                            e_i = edge_to_idx[e]
                            sros[-1].append(e_i)
                            resulting_embs.append(embedding_mat[e_i])
                    if len(resulting_embs):
                        resulting_embs = torch.stack(resulting_embs).to(args.device)
                        score = F.pairwise_distance(pred, resulting_embs, p=2).cpu().numpy()
                        scores.append(score)
                    else:
                        scores.append(np.array([]))
        else:
            cdist_bs = 32
            for i in tqdm(list(range(0, len(val_preds), cdist_bs)), desc="per-question edge finding (all edges)"):
                score = torch.cdist(torch.stack(val_preds[i:i + cdist_bs]).cpu(), embedding_mat).numpy()
                closest_edges = score.argsort()[:, :args.top_n_facts]
                scores.extend(np.take_along_axis(score, closest_edges, 1))
                sros.extend(closest_edges.tolist())

    with open(output_file, 'wb') as f:
        pickle.dump(scores, f)
    with jsonlines.open(f'{args.save_model_path}/recall_sros_{split_name}_{split}_{pretrained_embs_name}{e4s}.jsonl',
                        'w') as f:
        f.write_all(sros)


def main_predict():
    args = get_argparse()

    dir_paths = [
        'save_model/recall',
    ]
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, path)):
                continue
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            print(params)
            args.output_dir = params['output_dir']
            args.bert_path = params['bert_path']
            args.save_model_path = args.output_dir
            if args.sentence_transformer_model is None:
                model = TripletModel(
                    bert_model=args.bert_path
                )
                save_model_path = os.path.join(args.output_dir, 'model_best.bin')
                state_dict = torch.load(save_model_path)
                model.load_state_dict(state_dict)
            else:
                model = SentenceTransformer(args.sentence_transformer_model)
            args.device = torch.device("cuda")

            model.cuda()

            if not args.do_qa_eval:
                model_predict(args,
                              BertDataset(get_dev_examples()),
                              BertDataset(list(get_all_facts_from_id().keys())),
                              'val',
                              model)
                model_predict(args,
                              BertDataset(get_test_examples()),
                              BertDataset(list(get_all_facts_from_id().keys())),
                              'test',
                              model)
                model_predict(args,
                              BertDataset(get_train_predict_examples()),
                              BertDataset(list(get_all_facts_from_id().keys())),
                              'train',
                              model)
            else:
                model_predict_qa(args,
                                 BertDataset(get_qa_examples(args.statement_path)),
                                 'qa',
                                 model)


def get_result_train():
    dir_paths = [
        'save_model/recall',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, path)):
                continue
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/recall_scores_train.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/recall_scores_train.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    print(merge_scores.shape)
    train_examples = get_train_predict_examples()
    val_dataset = BertDataset(train_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('data/predict_tem.txt', 'w')
    val_top_2000 = defaultdict(list)
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        indices = scores.argsort()[:2000]
        recall_subject_ids = [all_facts_keys[index] for index in indices]
        for recall_id in recall_subject_ids:
            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    score = eval_ndcg_train('data/predict_tem.txt')

    pd.to_pickle(val_top_2000, 'data/train_top_2000.pkl')


def get_result_val():
    dir_paths = [
        'save_model/recall',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, path)):
                continue
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/recall_scores_val.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/recall_scores_val.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    val_examples = get_dev_examples()
    val_dataset = BertDataset(val_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('data/val_predict_tem.txt', 'w')
    idtopositives = get_idtopositives_val()
    val_top_2000 = defaultdict(list)
    totals = 0
    preds_num = 0
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        indices = scores.argsort()[:2000]
        recall_subject_ids = [all_facts_keys[index] for index in indices]
        totals += len(idtopositives[query_id])
        for recall_id in recall_subject_ids:
            if recall_id in idtopositives[query_id]:
                preds_num += 1
            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    print(preds_num / totals)
    score = eval_ndcg('data/val_predict_tem.txt')

    pd.to_pickle(val_top_2000, 'data/val_top_2000.pkl')


def get_result_test():
    dir_paths = [
        'save_model/recall',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, path)):
                continue
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/recall_scores_test.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/recall_scores_test.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    val_examples = get_test_examples()
    val_dataset = BertDataset(val_examples)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    val_predict = open('data/test_predict2000.txt', 'w')
    val_top_2000 = defaultdict(list)
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        indices = scores.argsort()[:2000]
        recall_subject_ids = [all_facts_keys[index] for index in indices]
        for recall_id in recall_subject_ids:
            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    pd.to_pickle(val_top_2000, 'data/test_top_2000.pkl')


def get_result_qa(split, e4s):
    args = get_argparse()
    if args.edge_emb_path is None:
        return
    pretrained_embs_name = args.edge_emb_path.split("/")[-1].replace("_embedding_embs.pt", "")

    e4s = '' if e4s else '_all_edges'
    specific_qa_name = f'qa_{split}_{pretrained_embs_name}{e4s}'

    output_file = f'data/{specific_qa_name}_top_{args.top_n_facts}.pkl'
    if os.path.exists(output_file):
        return

    dir_paths = [
        'save_model/recall',
    ]

    merge_scores = {}
    # not tested for ensembling multiple models but should work
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, path)):
                continue
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))

            model_recall_scores_file = f'{params["output_dir"]}/recall_scores_{specific_qa_name}.pkl'

            if not os.path.isfile(model_recall_scores_file):
                continue

            with open(model_recall_scores_file, 'rb') as f:
                scores = pickle.load(f)

            with jsonlines.open(f'{params["output_dir"]}/recall_sros_{specific_qa_name}.jsonl') as f:
                sros_for_questions = list(f)
            for i, (sros, score_list) in tqdm(enumerate(zip(sros_for_questions, scores)),
                                              desc="processing scores from model", total=len(scores)):
                merge_scores[i] = defaultdict(list)
                for sro, score in zip(sros, score_list):
                    merge_scores[i][sro].append(score)

    if not len(merge_scores):
        return

    # merge_scores: question number -> triple id -> [score]

    merge_scores = list(merge_scores.items())
    # ensure ascending question order
    merge_scores.sort(key=lambda x: x[0])
    merge_scores = [list(question_data.items()) for question_id, question_data in merge_scores]

    qa_top_2000 = {}
    for i, ids_with_scores in tqdm(enumerate(merge_scores),
                                   desc=f"processing top {args.top_n_facts} from questions",
                                   total=len(merge_scores)):
        # can't use defaultdict as some qa pairs have no scores but we still need an entry
        qa_top_2000[i] = []
        if not len(ids_with_scores):
            continue
        triple_ids, scores = zip(*ids_with_scores)
        scores = np.asarray(scores).mean(axis=1)
        indices = scores.argsort()[:args.top_n_facts]
        fact_ids_from_indices = [triple_ids[j] for j in indices]
        for recall_id in fact_ids_from_indices:
            qa_top_2000[i].append(recall_id)

    with open(output_file, 'wb') as f:
        pickle.dump(qa_top_2000, f)


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2021, type=int,
                        help="")
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument("--per_gpu_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training or evaluation.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training or evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Bert.")
    parser.add_argument("--lr", default=5e-4, type=float,
                        help="The initial learning rate")
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument("--do_eval", default=True, type=bool, )

    parser.add_argument("--do_adv", default=True, type=bool)
    parser.add_argument('--dropout_num', default=1, type=int)
    parser.add_argument('--num_hidden_layers', default=1, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--dropout_prob1', default=0.2, type=float)
    parser.add_argument('--dropout_prob2', default=0.1, type=float)

    parser.add_argument("--output_dir", default='', type=str)
    parser.add_argument("--bert_path", default='', type=str, )

    parser.add_argument("--do_qa_eval", action="store_true")
    parser.add_argument("--statement_path", type=str, )
    parser.add_argument("--edges_for_statements", type=str, )
    parser.add_argument("--edge_emb_path", type=str, )
    parser.add_argument("--edge_emb_mapping_path", type=str,
                        help="make sure this matches the model (deepblue vs sbert)")
    parser.add_argument("--cpnet_folder", type=str, help="should be removed, not used anymore")
    parser.add_argument("--fact-bs", default=64, type=int)
    parser.add_argument("--top-n-facts", default=2000, type=int)
    parser.add_argument("--sentence-transformer-model", type=str,
                        help="if using, make sure edge embeddings are also from sbert")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main_predict()
    # get_result_val()
    # get_result_test()
    # get_result_train()
    for split in ['train', 'test', 'dev']:
        for e4s in [True, False]:
            get_result_qa(split, e4s)
