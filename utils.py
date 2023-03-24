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

        if self.args.noise_prob > 0:
            self.vocab_pool = list(set(range(len(tokenizer))) - set(tokenizer.all_special_ids))

    def get_predict_dataloader(self):
        predict_dataset = KeyDataset(self.predict_data)
        predict_dataloader = DataLoader(predict_dataset,
                                        batch_size=self.args.batch_size * 2,
                                        collate_fn=self.predict_collate)
        return predict_dataloader

    def read_file(self, file):
        return [json.loads(x) for x in open(file, encoding='utf-8')]

    def encode_src(self, src):
        res = self.tokenizer(src,
                             padding=True,
                             return_tensors='pt',
                             max_length=self.args.max_source_length,
                             truncation='longest_first',
                             return_attention_mask=True,
                             return_token_type_ids=False)
        return res

    def train_collate(self, batch):
        if isinstance(batch[0], list):
            batch = batch[0]  # max_token_dataset
        src = [b['src'] for b in batch]
        tgt = [b['tgt'] for b in batch]

        src_tokenized = self.encode_src(src)
        with self.tokenizer.as_target_tokenizer():
            tgt_tokenized = self.tokenizer(
                tgt,
                max_length=self.args.max_target_length,
                padding=True,
                return_tensors='pt',
                truncation='longest_first')

        decoder_attention_mask = tgt_tokenized['attention_mask'][:, :-1]
        decoder_input_ids = tgt_tokenized['input_ids'][:, :-1]

        labels = tgt_tokenized['input_ids'][:, 1:].clone()
        labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)

        if self.args.noise_prob > 0:
            noise_indices = torch.rand_like(labels) < self.args.noise_prob
            noise_indices = noise_indices & (decoder_input_ids != self.tokenizer.bos_token_id) \
                            & (labels != self.tokenizer.eos_token_id) & decoder_attention_mask.bool()
            noise_inp = np.random.choice(self.vocab_pool, decoder_input_ids.shape)
            decoder_input_ids = torch.where(noise_indices, noise_inp, decoder_input_ids)

        res = {'input_ids': src_tokenized['input_ids'],
               'attention_mask': src_tokenized['attention_mask'],
               'decoder_input_ids': decoder_input_ids,
               'decoder_attention_mask': decoder_attention_mask,
               'labels': labels}
        return res

    def dev_collate(self, batch):
        return self.train_collate(batch)

    def predict_collate(self, batch):
        src = [x['src'] for x in batch]
        return self.encode_src(src)

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
                                              num_workers=self.args.num_workers,
                                              shuffle=True)
                dev_dataloader = DataLoader(dev_dataset,
                                            batch_size=self.args.batch_size * 2,
                                            collate_fn=self.dev_collate)
                ret['train'].append(train_dataloader)
                ret['dev'].append(dev_dataloader)
        else:
            if self.args.kfold == 1 and self.dev_data is None:
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
                                          num_workers=self.args.num_workers, shuffle=True)
            dev_dataloader = DataLoader(dev_dataset,
                                        batch_size=self.args.batch_size * 2,
                                        collate_fn=self.dev_collate)
            ret['train'].append(train_dataloader)
            ret['dev'].append(dev_dataloader)

        return ret


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


def compute_rouge(label, pred, weights=None):
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
    return {
        'rouge': sum(s * w for s, w in zip(scores, weights)),
        'rouge-1': scores[0], 'rouge-2': scores[1], 'rouge-l': scores[2]
    }


def ce_loss(logits, labels, is_prob=False, eps=0):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    if not is_prob:
        loss = F.cross_entropy(logits, labels, label_smoothing=eps)
    else:
        lprob = (logits + 1e-9).log()
        loss = F.nll_loss(lprob, labels)
    return loss


def kl_loss(logtis, logits2, mask):
    prob1 = F.softmax(logtis, -1)
    prob2 = F.softmax(logits2, -1)
    lprob1 = prob1.log()
    lprob2 = prob2.log()
    loss1 = F.kl_div(lprob1, lprob2, reduction='none')
    loss2 = F.kl_div(lprob2, lprob1, reduction='none')
    mask = (mask == 0).bool()
    loss1 = loss1.masked_fill_(mask, 0.0).sum()
    loss2 = loss2.masked_fill_(mask, 0.0).sum()
    loss = (loss1 + loss2) / 2

    return loss


# def mask_select(inputs, mask):
#     input_dim = inputs.ndim
#     mask_dim = mask.ndim
#     mask = mask.reshape(-1).bool()
#     if input_dim > mask_dim:
#         inputs = inputs.reshape((int(mask.size(-1)), -1))[mask]
#     else:
#         inputs = inputs.reshape(-1)[mask]
#     return inputs


# def copy_loss(inputs, targets, mask, eps=1e-6):
#     mask = mask[:, 1:]
#     inputs = inputs[:, :-1]
#     targets = targets[:, 1:]
#     inputs = mask_select(inputs, mask)
#     targets = mask_select(targets, mask)
#     log_preds = (inputs + eps).log()
#     loss = F.nll_loss(log_preds, targets)
#     return loss
#
#
# def ce_loss(inputs, targets, mask):
#     mask = mask[:, 1:]
#     inputs = inputs[:, :-1]
#     targets = targets[:, 1:]
#     inputs = mask_select(inputs, mask)
#     targets = mask_select(targets, mask)
#     loss = F.cross_entropy(inputs, targets)
#     return loss


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
