import argparse
import json
import logging
import os
import pickle
import random
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset import *
from model.models import TripletModel
from recall_predict import load_pretrained_fact_embs, encode_qa_dataset
from utils import name_to_split

logger = logging.getLogger()
from evaluate import eval_ndcg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_test_predict(args, model):
    # if os.path.isfile(args.save_model_path + '/sort_scores_test.pkl'):
    #     return
    test_examples = get_test_examples()
    test_dataset = BertDataset(test_examples)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    facts_preds = []
    fact_loader = DataLoader(all_facts_keys, batch_size=args.batch_size,
                             collate_fn=DataCollatorForExplanation(tokenizer))
    fact_loader = tqdm(fact_loader)
    for batch_index, inputs in enumerate(fact_loader):  # 调用模型计算facts表示
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        facts_preds.append(anchor_out)
    facts_preds = torch.cat(facts_preds, dim=0)

    facts_preds_dict = {}
    for i, j in zip(all_facts_keys, facts_preds):
        facts_preds_dict[i] = j

    dev_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            collate_fn=DataCollatorForTest(tokenizer, mode='test'))
    dev_loader = tqdm(dev_loader)
    val_preds = []
    for batch_index, inputs in enumerate(dev_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        val_preds.extend(anchor_out)
    test_recall_top2000 = pd.read_pickle('data/test_top_2000.pkl')
    scores = []
    for index, (pred, data) in enumerate(zip(val_preds, test_dataset)):
        query_id = test_dataset.__getitem__(index)
        recall_ids = test_recall_top2000[query_id][:2000]
        facts_pred_recall = [facts_preds_dict[rid] for rid in recall_ids]
        facts_pred_recall = torch.stack(facts_pred_recall, dim=0)
        score = F.pairwise_distance(pred, facts_pred_recall, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    print(scores.shape)
    pd.to_pickle(scores, args.save_model_path + '/sort_scores_test.pkl')


def model_val_predict(args, model):
    # if os.path.isfile(args.save_model_path + '/sort_scores_val.pkl'):
    #     return
    val_examples = get_dev_examples()
    val_dataset = BertDataset(val_examples)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    all_facts_keys = list(get_all_facts_from_id().keys())
    all_facts_keys = BertDataset(all_facts_keys)
    facts_preds = []
    fact_loader = DataLoader(all_facts_keys, batch_size=args.batch_size,
                             collate_fn=DataCollatorForExplanation(tokenizer))
    fact_loader = tqdm(fact_loader)
    for batch_index, inputs in enumerate(fact_loader):  # 调用模型计算facts表示
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        facts_preds.append(anchor_out)
    facts_preds = torch.cat(facts_preds, dim=0)

    facts_preds_dict = {}
    for i, j in zip(all_facts_keys, facts_preds):
        facts_preds_dict[i] = j
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=DataCollatorForTest(tokenizer, mode='val'))
    dev_loader = tqdm(dev_loader)
    val_preds = []
    for batch_index, inputs in enumerate(dev_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        anchor_out = model.BERTModel(**inputs).detach()
        val_preds.extend(anchor_out)
    scores = []
    val_recall_top2000 = pd.read_pickle('data/val_top_2000.pkl')

    for index, (pred, data) in enumerate(zip(val_preds, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        recall_ids = val_recall_top2000[query_id][:2000]
        facts_pred_recall = [facts_preds_dict[rid] for rid in recall_ids]
        facts_pred_recall = torch.stack(facts_pred_recall, dim=0)
        score = F.pairwise_distance(pred, facts_pred_recall, p=2).cpu().numpy()
        scores.append(score)
    scores = np.array(scores)
    print(scores.shape)
    pd.to_pickle(scores, args.save_model_path + '/sort_scores_val.pkl')


def model_qa_predict_using_recall_answers(args, model):
    statement_file = args.statement_path.split("/")[-1]
    split = name_to_split(statement_file)

    pretrained_embs_name = args.edge_emb_path.split("/")[-1].replace("_embedding_embs.pt", "")

    e4s = '' if not args.comparing_entire_graph else '_all_edges'
    specific_qa_name = f'qa_{split}_{pretrained_embs_name}{e4s}'

    output_file = f'{args.save_model_path}/sort_scores_{specific_qa_name}.pkl'
    if os.path.isfile(output_file):
        return

    recall_top_2000_f = f'data/{specific_qa_name}_top_{args.top_n_facts}.pkl'
    if not os.path.exists(recall_top_2000_f):
        return

    qa_dataset = BertDataset(get_qa_examples(args.statement_path))
    with torch.no_grad():
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

        nodes_to_edges, edge_to_idx, embedding_mat = load_pretrained_fact_embs(args)

        val_preds = encode_qa_dataset(qa_dataset, model, tokenizer, args, 'qa')
        if args.save_qa_encodings and not os.path.exists(f"data/qa_{split}_{pretrained_embs_name}_question_encodings.pt"):
            torch.save(torch.stack(val_preds).cpu(), f"data/qa_{split}_{pretrained_embs_name}_question_encodings.pt")

        scores = []
        with open(recall_top_2000_f, 'rb') as f:
            val_recall_top2000 = pickle.load(f)

        for i, pred in tqdm(enumerate(val_preds), total=len(val_preds), desc="scoring facts for questions"):
            recall_ids = val_recall_top2000[i][:args.top_n_facts]
            if not len(recall_ids):
                scores.append(np.array([]))
            else:
                facts_pred_recall = [embedding_mat[rid] for rid in recall_ids]
                facts_pred_recall = torch.stack(facts_pred_recall, dim=0).to(args.device)
                score = F.pairwise_distance(pred, facts_pred_recall, p=2).cpu().numpy()
                scores.append(score)

        with open(output_file, 'wb') as f:
            pickle.dump(scores, f)


def model_qa_predict(args, model):
    statement_file = args.statement_path.split("/")[-1]
    split = name_to_split(statement_file)

    pretrained_embs_name = args.edge_emb_path.split("/")[-1].replace("_embedding_embs.pt", "")

    e4s = '' if not args.comparing_entire_graph else '_all_edges'
    specific_qa_name = f'{args.dataset}_{split}_{pretrained_embs_name}{e4s}'

    output_file = f'{args.save_model_path}/sort_scores_{specific_qa_name}.pkl'
    if os.path.isfile(output_file):
        return

    qa_dataset = BertDataset(get_qa_examples(args.statement_path))
    with torch.no_grad():
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

        nodes_to_edges, edge_to_idx, embedding_mat = load_pretrained_fact_embs(args)

        val_preds = encode_qa_dataset(qa_dataset, model, tokenizer, args, 'qa')
        if args.save_qa_encodings and not os.path.exists(f"data/{specific_qa_name}_question_encodings.pt"):
            torch.save(torch.stack(val_preds).cpu(), f"data/{specific_qa_name}_question_encodings.pt")

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
        with jsonlines.open(f'{args.save_model_path}/sort_sros_{specific_qa_name}.jsonl', 'w') as f:
            f.write_all(sros)


def main_predict():
    args = get_argparse()
    dir_paths = [
        'save_model/sort',
    ]

    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            args.output_dir = params['output_dir']
            args.bert_path = params['bert_path']
            args.save_model_path = args.output_dir
            args.device = torch.device("cuda")

            if args.sentence_transformer_model is None:
                model = TripletModel(
                    bert_model=args.bert_path,
                )
                save_model_path = os.path.join(args.output_dir, 'model_best.bin')
                if not os.path.isfile(save_model_path):
                    continue
                state_dict = torch.load(save_model_path)
                model.load_state_dict(state_dict)
            else:
                model = SentenceTransformer(args.sentence_transformer_model)

            model.cuda()

            if args.statement_path is None:
                model_val_predict(args, model)
                model_test_predict(args, model)
            else:
                model_qa_predict(args, model)


def get_result_val():
    dir_paths = [
        'save_model/sort',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            if not os.path.isfile(params['output_dir'] + '/sort_scores_val.pkl'):
                continue
            scores = pd.read_pickle(params['output_dir'] + '/sort_scores_val.pkl')
            merge_scores.append(scores)
    print(len(merge_scores))
    merge_scores = np.mean(merge_scores, axis=0)
    print(merge_scores.shape)
    val_examples = get_dev_examples()
    val_dataset = BertDataset(val_examples)
    val_predict = open('data/val_predict_tem.txt', 'w')
    val_top_2000 = defaultdict(list)
    val_recall_top2000 = pd.read_pickle('data/val_top_2000.pkl')
    for index, (scores, data) in enumerate(zip(merge_scores, val_dataset)):
        query_id = val_dataset.__getitem__(index)
        recall_ids = val_recall_top2000[query_id]

        indices = scores.argsort()
        recall_subject_ids = [recall_ids[index] for index in indices]
        for recall_id in recall_subject_ids:
            val_top_2000[query_id].append(recall_id)
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()
    score = eval_ndcg('data/val_predict_tem.txt')
    print('score', score)


def get_result_test():
    dir_paths = [
        'save_model/sort',
    ]
    merge_scores = []
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))
            scores = pd.read_pickle(params['output_dir'] + '/sort_scores_test.pkl')
            merge_scores.append(scores)
    merge_scores = np.mean(merge_scores, axis=0)
    test_examples = get_test_examples()
    test_dataset = BertDataset(test_examples)
    val_predict = open('result/predict.txt', 'w')

    test_recall_top2000 = pd.read_pickle('data/test_top_2000.pkl')

    for index, (scores, data) in enumerate(zip(merge_scores, test_dataset)):
        query_id = test_dataset.__getitem__(index)

        recall_ids = test_recall_top2000[query_id]

        indices = scores.argsort()
        recall_subject_ids = [recall_ids[index] for index in indices]

        for recall_id in recall_subject_ids[:2000]:
            val_predict.write('{}\t{}\n'.format(query_id, recall_id))
    val_predict.close()


def get_result_qa(split, e4s):
    dir_paths = [
        'save_model/sort',
    ]
    args = get_argparse()
    pretrained_embs_name = args.edge_emb_path.split("/")[-1].replace("_embedding_embs.pt", "")

    e4s = '' if e4s else '_all_edges'
    specific_qa_name = f'{args.dataset}_{split}_{pretrained_embs_name}{e4s}'
    result_dir = f'result/{pretrained_embs_name}{e4s}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file = f'{result_dir}/{specific_qa_name}.pkl'
    if os.path.exists(result_file):
        return

    merge_scores = {}
    for model_path in dir_paths:
        for path in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, path)):
                continue

            args_path = os.path.join(model_path, path, 'args.json')
            params = json.load(open(args_path, 'r'))

            model_sort_scores_file = f'{params["output_dir"]}/sort_scores_{specific_qa_name}.pkl'
            model_sort_sros_file = f'{params["output_dir"]}/sort_sros_{specific_qa_name}.jsonl'

            if not os.path.isfile(model_sort_scores_file) or not os.path.isfile(model_sort_sros_file):
                continue
            with open(model_sort_scores_file, 'rb') as f:
                scores = pickle.load(f)
            with jsonlines.open(model_sort_sros_file) as f:
                sros = list(f)

            for i, score_list in tqdm(enumerate(scores), desc="processing scores from model", total=len(scores)):
                merge_scores[i] = defaultdict(list)
                edge_ids = sros[i]
                for sro, score in zip(edge_ids, score_list):
                    merge_scores[i][sro].append(score)

    if not len(merge_scores):
        return

    merge_scores = list(merge_scores.items())
    # ensure ascending question order
    merge_scores.sort(key=lambda x: x[0])
    # each item is [(edge_id, [scores])]
    merge_scores = [list(question_data.items()) for question_id, question_data in merge_scores]

    final_outputs = []
    for i, question_scores in tqdm(enumerate(merge_scores), total=len(merge_scores), desc="final question processing"):
        if not len(question_scores):
            final_outputs.append(([], []))
        else:
            question_scores = [(edge_id, np.mean(score)) for edge_id, score in question_scores]
            edge_ids, scores = zip(*question_scores)
            scores = np.asarray(scores)
            indices = scores.argsort()
            scores = scores[indices]
            reordered_edge_ids = [edge_ids[j] for j in indices]
            final_outputs.append((reordered_edge_ids, scores))
    with open(result_file, 'wb') as f:
        pickle.dump(final_outputs, f)


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

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--bert_path", type=str)

    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--edge_emb_path", type=str, )
    parser.add_argument("--dataset", default='qa')
    parser.add_argument("--edge_emb_mapping_path", type=str,
                        help="shouldn't be sbert")
    parser.add_argument("--edges_for_statements", type=str, )
    parser.add_argument("--top-n-facts", default=2000, type=int)
    parser.add_argument("--statement_path", type=str, )
    parser.add_argument("--comparing-entire-graph", action='store_true')
    parser.add_argument("--save-qa-encodings", action="store_true")
    parser.add_argument("--sentence-transformer-model", type=str,
                        help="")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main_predict()
    # get_result_val()
    # get_result_test()
    for split in ['train', 'test', 'dev']:
        for e4s in [True, False]:
            get_result_qa(split, e4s)
