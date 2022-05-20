import os
import json
import logging

import jsonlines
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torch.utils.data.dataset import Dataset
import random
import torch


__all__ = [
    'BertDataset', 'init_logger', 'get_all_facts_from_id', 'get_idtopositives_val',
    'DataCollatorForTrainSort', 'DataCollatorForTrain', 'DataCollatorForTest', 'DataCollatorForExplanation',
    'get_train_predict_examples', 'get_train_examples', 'get_dev_examples', 'get_test_examples', 'get_qa_examples'
]


def init_logger(log_file=None, log_file_level=logging.INFO):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def read_explanations(path: str, file_name):
    header = []
    uid = None
    file_name = file_name.split('.')[0].lower().replace('-', ' ')
    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)

    if not uid or len(df) == 0:
        return []
    explanations = df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()
    explanations_dict = {explanation[0]: (file_name, explanation[1]) for explanation in explanations}
    return explanations_dict


def get_all_facts_from_id():
    explanations_dict = {}
    for path, _, files in os.walk('data/tables'):
        for file in files:
            ex_dict = read_explanations(os.path.join(path, file), file)
            for k in ex_dict:
                explanations_dict[k] = ex_dict[k]
    return explanations_dict


def get_all_facts_from_file_name():
    explanations_dict = defaultdict(list)
    for path, _, files in os.walk('data/tables'):
        for file in tqdm(files, desc="Loading files"):
            ex_dict = read_explanations(os.path.join(path, file), file)
            for k in ex_dict:
                file_name, t = ex_dict[k]
                explanations_dict[file_name].append(k)
    return explanations_dict


def _get_idtotext(split):
    idtotext = {}
    path = f'data/wt-expert-ratings.{split}.json'
    with open(path, 'rb') as f:
        questions_file = json.load(f)
    for ranking_problem in questions_file['rankingProblems']:
        question_id = ranking_problem['qid']
        question_text = ranking_problem['questionText']
        answer_text = ranking_problem['answerText']
        idtotext[question_id] = (question_text, answer_text)
    return idtotext


def get_train_idtotext():
    return _get_idtotext('train')


def get_test_idtotext():
    return _get_idtotext('test')


def get_dev_idtotext():
    return _get_idtotext('dev')


def get_qa_idtotext(statement_path):
    idtotext = {}
    with jsonlines.open(statement_path) as f:
        for obj in f:
            for choice in obj['question']['choices']:
                idtotext[len(idtotext)] = (obj['question']['stem'], choice['text'])
    return idtotext


def get_idtopositives():
    idtopositives = {}
    path = 'data/wt-expert-ratings.train.json'

    with open(path, 'rb') as f:
        questions_file = json.load(f)
    for ranking_problem in questions_file['rankingProblems']:
        question_id = ranking_problem['qid']
        positives = ranking_problem['documents']
        positive_set = set()
        for positive in positives:
            positive_set.add(positive['uuid'])
        idtopositives[question_id] = positive_set
    return idtopositives


def get_idtopositives_val():
    idtopositives = {}
    path = 'data/wt-expert-ratings.dev.json'

    with open(path, 'rb') as f:
        questions_file = json.load(f)
    for ranking_problem in questions_file['rankingProblems']:
        question_id = ranking_problem['qid']
        positives = ranking_problem['documents']
        positive_set = set()
        for positive in positives:
            positive_set.add(positive['uuid'])
        idtopositives[question_id] = positive_set
    return idtopositives


def get_train_examples(relevance_int=5):
    examples = []
    path = 'data/wt-expert-ratings.train.json'
    with open(path, 'rb') as f:
        questions_file = json.load(f)
    for ranking_problem in questions_file['rankingProblems']:
        question_id = ranking_problem['qid']
        positives = ranking_problem['documents']
        for positive in positives:
            if positive['relevance'] > relevance_int:
                positive_id = positive['uuid']
                examples.append((question_id, positive_id))
    return examples


def _load_predict_examples(split):
    path = f'data/wt-expert-ratings.{split}.json'
    with open(path, 'rb') as f:
        questions_file = json.load(f)
    examples = []
    for ranking_problem in questions_file['rankingProblems']:
        question_id = ranking_problem['qid']
        examples.append(question_id)
    return examples


def get_train_predict_examples():
    return _load_predict_examples('train')


def get_dev_examples():
    return _load_predict_examples('dev')


def get_test_examples():
    return _load_predict_examples('test')


def get_qa_examples(statement_path):
    examples = []
    with jsonlines.open(statement_path) as f:
        for obj in f:
            for _ in obj['question']['choices']:
                examples.append(len(examples))
    return examples


class BertDataset(Dataset):

    def __init__(
            self,
            examples
    ):
        self.features = examples

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


def get_negative(positive_id, positive_ids, bacth_positives, all_facts_id, all_facts_file):
    neg_id = random.choice(list(all_facts_id.keys()))
    for i in range(5):
        neg_id = random.choice(list(all_facts_id.keys()))
        if neg_id not in positive_ids:
            break
    neg_id1, neg_id2 = neg_id, neg_id
    file_name = all_facts_id[positive_id][0]
    candidate_file = list(all_facts_file[file_name])
    for i in range(5):
        neg_id1 = random.choice(candidate_file)
        if neg_id1 not in positive_ids:
            break
    for i in range(5):
        neg_id2 = random.choice(bacth_positives)
        if neg_id2 not in positive_ids:
            break
    return [neg_id, neg_id1, neg_id2]


class DataCollatorForTrain:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.idtopositives = get_idtopositives()
        self.idtotext = get_train_idtotext()
        self.all_facts_id = get_all_facts_from_id()
        self.all_facts_file = get_all_facts_from_file_name()

    def __call__(self, features):
        batch = {}
        bacth_positives = []
        for feature in features:
            bacth_positives.append(feature[1])
        new_features = []
        for feature in features:
            question_id = feature[0]
            positive_id = feature[1]
            positive_ids = self.idtopositives[question_id]
            negatives = get_negative(positive_id, positive_ids, bacth_positives, self.all_facts_id, self.all_facts_file)

            for neg in negatives:
                new_features.append((question_id, positive_id, neg))
        anchor_ids = []
        positive_ids = []
        negative_ids = []

        for feature in new_features:
            anchor_ids.append(' <answer> '.join(self.idtotext[feature[0]]))
            positive_ids.append(' '.join(self.all_facts_id[feature[1]]))
            negative_ids.append(' '.join(self.all_facts_id[feature[2]]))

        anchor_batch = self.tokenizer.batch_encode_plus(
            anchor_ids,
            max_length=256,
            padding=True,
            truncation=True,
        )
        batch['anchor_ids'] = torch.tensor(anchor_batch['input_ids'], dtype=torch.long)
        batch['anchor_mask'] = torch.tensor(anchor_batch['attention_mask'], dtype=torch.long)

        positive_batch = self.tokenizer.batch_encode_plus(
            positive_ids,
            max_length=256,
            padding=True,
            truncation=True,
        )
        batch['positive_ids'] = torch.tensor(positive_batch['input_ids'], dtype=torch.long)
        batch['positive_mask'] = torch.tensor(positive_batch['attention_mask'], dtype=torch.long)

        negative_batch = self.tokenizer.batch_encode_plus(
            negative_ids,
            max_length=256,
            padding=True,
            truncation=True,
        )
        batch['negative_ids'] = torch.tensor(negative_batch['input_ids'], dtype=torch.long)
        batch['negative_mask'] = torch.tensor(negative_batch['attention_mask'], dtype=torch.long)

        return batch


class DataCollatorForTest:
    def __init__(self, tokenizer, args=None, mode='val'):
        self.tokenizer = tokenizer
        if mode == 'val':
            self.idtotext = get_dev_idtotext()
        elif mode == 'train':
            self.idtotext = get_train_idtotext()
        elif mode == 'test':
            self.idtotext = get_test_idtotext()
        elif mode == 'qa':
            assert args is not None
            self.idtotext = get_qa_idtotext(args.statement_path)
        else:
            raise ValueError()
        self.return_tokenized = args is None or args.sentence_transformer_model is None

    def __call__(self, features):
        batch = {}
        anchor_ids = []

        answer_joiner = ' <answer> ' if self.return_tokenized else ' is answered by '

        print("FEATURES")
        print(features)
        for feature in features:
            anchor_ids.append(answer_joiner.join(self.idtotext[feature]))

        if self.return_tokenized:
            anchor_batch = self.tokenizer.batch_encode_plus(
                anchor_ids,
                max_length=256,
                padding=True,
                truncation=True,
            )
            batch['input_ids'] = torch.tensor(anchor_batch['input_ids'], dtype=torch.long)
            batch['attention_mask'] = torch.tensor(anchor_batch['attention_mask'], dtype=torch.long)
            return batch
        else:
            return anchor_ids


class DataCollatorForExplanation:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.all_facts_id = get_all_facts_from_id()

    def __call__(self, features):
        batch = {}
        explanations = []
        for feature in features:
            explanations.append(' '.join(self.all_facts_id[feature]))
        anchor_batch = self.tokenizer.batch_encode_plus(
            explanations,
            max_length=256,
            padding=True,
            truncation=True,
        )
        batch['input_ids'] = torch.tensor(anchor_batch['input_ids'], dtype=torch.long)
        batch['attention_mask'] = torch.tensor(anchor_batch['attention_mask'], dtype=torch.long)
        return batch


def get_negative_sort(positive_id, positive_ids, bacth_positives, recall_top2000):
    neg_id = random.choice(recall_top2000)
    for i in range(5):
        neg_id = random.choice(recall_top2000)
        if neg_id not in positive_ids:
            break
    neg_id1, neg_id2 = neg_id, neg_id

    top100 = recall_top2000[:100]
    for i in range(5):
        neg_id1 = random.choice(top100)
        if neg_id1 not in positive_ids:
            break
    for i in range(5):
        neg_id2 = random.choice(bacth_positives)
        if neg_id2 not in positive_ids:
            break
    return [neg_id, neg_id1, neg_id2]


class DataCollatorForTrainSort:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.idtopositives = get_idtopositives()
        self.idtotext = get_train_idtotext()
        self.recall_top_2000 = pd.read_pickle('data/train_top_2000.pkl')
        self.all_facts_id = get_all_facts_from_id()
        self.all_facts_file = get_all_facts_from_file_name()

    def __call__(self, features):
        batch = {}
        bacth_positives = []
        for feature in features:
            bacth_positives.append(feature[1])
        new_features = []
        for feature in features:
            question_id = feature[0]
            positive_id = feature[1]
            positive_ids = self.idtopositives[question_id]
            negatives = get_negative_sort(positive_id, positive_ids, bacth_positives, self.recall_top_2000[question_id])
            for neg in negatives:
                new_features.append((question_id, positive_id, neg))
        anchor_ids = []
        positive_ids = []
        negative_ids = []

        for feature in new_features:
            anchor_ids.append(' <answer> '.join(self.idtotext[feature[0]]))
            positive_ids.append(' '.join(self.all_facts_id[feature[1]]))
            negative_ids.append(' '.join(self.all_facts_id[feature[2]]))

        anchor_batch = self.tokenizer.batch_encode_plus(
            anchor_ids,
            max_length=256,
            padding=True,
            truncation=True,
        )
        batch['anchor_ids'] = torch.tensor(anchor_batch['input_ids'], dtype=torch.long)
        batch['anchor_mask'] = torch.tensor(anchor_batch['attention_mask'], dtype=torch.long)

        positive_batch = self.tokenizer.batch_encode_plus(
            positive_ids,
            max_length=256,
            padding=True,
            truncation=True,
        )
        batch['positive_ids'] = torch.tensor(positive_batch['input_ids'], dtype=torch.long)
        batch['positive_mask'] = torch.tensor(positive_batch['attention_mask'], dtype=torch.long)

        negative_batch = self.tokenizer.batch_encode_plus(
            negative_ids,
            max_length=256,
            padding=True,
            truncation=True,
        )
        batch['negative_ids'] = torch.tensor(negative_batch['input_ids'], dtype=torch.long)
        batch['negative_mask'] = torch.tensor(negative_batch['attention_mask'], dtype=torch.long)

        return batch


if __name__ == '__main__':
    pass
