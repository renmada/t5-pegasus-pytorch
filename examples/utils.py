from transformers import BertTokenizer
from functools import partial
import json
import jieba
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import rouge
import re
from transformers import AdamW
import collections

rouge = rouge.Rouge()
smooth = SmoothingFunction().method1


class EncoderDecoderData:
    def __init__(self, args, tokenizer, ):
        self.train_data = self.read_file(args.train_file) if args.train_file else None
        self.dev_data = self.read_file(args.dev_file) if args.dev_file else None
        self.predict_data = self.read_file(args.predict_file) if args.predict_file else None
        self.args = args
        self.tokenizer = tokenizer

    def get_predict_dataloader(self):
        predict_dataset = KeyDataset(self.predict_data)
        predict_dataloader = DataLoader(predict_dataset, batch_size=self.args.batch_size * 2,
                                        collate_fn=self.predict_collate)
        return predict_dataloader

    def read_file(self, file):
        return [json.loads(x) for x in open(file, encoding='utf-8')]

    def train_collate(self, batch):
        source = [x['src'] for x in batch]
        target = [x['tgt'] for x in batch]
        res = self.tokenizer(source,
                             padding=True,
                             return_tensors='pt',
                             max_length=512,
                             truncation='longest_first',
                             return_attention_mask=True,
                             return_token_type_ids=False)

        target_features = self.tokenizer(target,
                                         padding=True,
                                         return_tensors='pt',
                                         max_length=150,
                                         truncation='longest_first',
                                         return_attention_mask=True,
                                         return_token_type_ids=False)
        res['decoder_attention_mask'] = target_features['attention_mask']
        res['labels'] = target_features['input_ids']
        if self.args.noise_prob == 0.:
            res['decoder_input_ids'] = target_features['input_ids']
        else:
            ids = target_features['input_ids'].clone()
            mask = res['decoder_attention_mask']
            noise_ids = torch.randint_like(ids, 1, 50000)
            noise_place = np.random.random(ids.shape) < self.args.noise_prob
            noise_place = torch.from_numpy(noise_place) & mask.bool()
            ids = torch.where(noise_place, noise_ids, ids)
            res['decoder_input_ids'] = ids
        return res

    def dev_collate(self, batch):
        return self.train_collate(batch)

    def predict_collate(self, batch):
        source = [x['src'] for x in batch]
        ids = [x['id'] for x in batch]
        res = self.tokenizer(source,
                             padding=True,
                             return_tensors='pt',
                             max_length=self.args.max_source_length,
                             return_attention_mask=True,
                             return_token_type_ids=False,
                             truncation='longest_first')
        res['id'] = torch.tensor(list(map(int, ids)))
        return res

    def get_dataloader(self):
        ret = {'train': [], 'dev': []}
        base_dataset = KeyDataset(self.train_data)
        if self.args.kfold > 1:
            from sklearn.model_selection import KFold
            for train_idx, dev_idx in KFold(n_splits=self.args.kfold, shuffle=True,
                                            random_state=self.args.seed).split(range(len(self.train_data))):
                train_dataset = Subset(base_dataset, train_idx)
                dev_dataset = Subset(base_dataset, dev_idx)
                train_dataloader = DataLoader(train_dataset,
                                              batch_size=self.args.batch_size,
                                              collate_fn=self.train_collate,
                                              num_workers=self.args.num_works,
                                              shuffle=True)
                dev_dataloader = DataLoader(dev_dataset,
                                            batch_size=self.args.batch_size * 2,
                                            collate_fn=self.dev_collate)
                ret['train'].append(train_dataloader)
                ret['dev'].append(dev_dataloader)
        else:
            if self.args.kfold == 1:
                from sklearn.model_selection import train_test_split
                train_idx, dev_idx = train_test_split(range(len(self.train_data)),
                                                      test_size=0.2,
                                                      random_state=self.args.seed)
                train_dataset = Subset(base_dataset, train_idx)
                dev_dataset = Subset(base_dataset, dev_idx)
            else:
                assert self.dev_data is not None, 'When no kfold, dev data must be targeted'
                train_dataset = base_dataset
                dev_dataset = KeyDataset(self.dev_data)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.args.batch_size,
                                          collate_fn=self.train_collate,
                                          num_workers=self.args.num_works, shuffle=True)
            dev_dataloader = DataLoader(dev_dataset,
                                        batch_size=self.args.batch_size * 2,
                                        collate_fn=self.dev_collate)
            ret['train'].append(train_dataloader)
            ret['dev'].append(dev_dataloader)

        return ret


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def compute_bleu(label, pred, weights=None):
    weights = weights or (0.25, 0.25, 0.25, 0.25)

    return np.mean([sentence_bleu(references=[list(a)], hypothesis=list(b), smoothing_function=smooth, weights=weights)
                    for a, b in zip(label, pred)])


def compute_rouge(label, pred, weights=None, mode='weighted'):
    weights = weights or (0.2, 0.4, 0.4)
    if isinstance(label, str):
        label = [label]
    if isinstance(pred, str):
        pred = [pred]
    label = [' '.join(x) for x in label]
    pred = [' '.join(x) for x in pred]

    def _compute_rouge(label, pred):
        try:
            scores = rouge.get_scores(hyps=label, refs=pred)[0]
            scores = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
        except ValueError:
            scores = [0, 0, 0]
        return scores

    scores = np.mean([_compute_rouge(*x) for x in zip(label, pred)], axis=0)
    if mode == 'weighted':
        return {'rouge': sum(s * w for s, w in zip(scores, weights))}
    elif mode == '1':
        return {'rouge-1': scores[0]}
    elif mode == '2':
        return {'rouge-2':scores[1]}
    elif mode == 'l':
        return {'rouge-l': scores[2]}
    elif mode == 'all':
        return {'rouge-1': scores[0], 'rouge-2':scores[1], 'rouge-l': scores[2]}


def mask_select(inputs, mask):
    input_dim = inputs.ndim
    mask_dim = mask.ndim
    mask = mask.reshape(-1).bool()
    if input_dim > mask_dim:
        inputs = inputs.reshape((int(mask.size(-1)), -1))[mask]
    else:
        inputs = inputs.reshape(-1)[mask]
    return inputs


def copy_loss(inputs, targets, mask, eps=1e-6):
    mask = mask[:, 1:]
    inputs = inputs[:, :-1]
    targets = targets[:, 1:]
    inputs = mask_select(inputs, mask)
    targets = mask_select(targets, mask)
    log_preds = (inputs + eps).log()
    loss = F.nll_loss(log_preds, targets)
    return loss

def ce_loss(inputs, targets, mask):
    mask = mask[:, 1:]
    inputs = inputs[:, :-1]
    targets = targets[:, 1:]
    inputs = mask_select(inputs, mask)
    targets = mask_select(targets, mask)
    loss = F.cross_entropy(inputs, targets)
    return loss

def create_optimizer(model, lr, weight_decay, custom_lr=None):
    no_decay = 'bias|norm'
    params = collections.defaultdict(list)
    custom_lr = custom_lr or dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        in_custom = False
        for custom_name, _ in custom_lr.items():
            if custom_name in name:
                if re.search(no_decay, name.lower()):
                    params[custom_name].append(param)
                else:
                    params[custom_name + '_decay'].append(param)
                in_custom = True
                break
        if not in_custom:
            if re.search(no_decay, name.lower()):
                params['normal'].append(param)
            else:
                params['normal_decay'].append(param)

    optimizer_grouped_parameters = []
    for k, v in params.items():
        param_lr = custom_lr.get(k.split('_')[0], lr)
        decay = weight_decay if 'decay' in k else 0.0
        optimizer_grouped_parameters.append({'params': v, 'weight_decay': decay, 'lr': param_lr}, )

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer
