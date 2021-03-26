from bert4torch.model import *
import rouge
from transformers import MT5ForConditionalGeneration
import jieba
from transformers import BertTokenizer
from torch._six import container_abcs, string_classes, int_classes
import torch
from torch.utils.data import DataLoader, Dataset
import re

rouge = rouge.Rouge()


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars

            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)

        return default_collate([default_collate(elem) for elem in batch])

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append((title, content))
    return D


train_data = load_data('/root/csl/train.tsv')
valid_data = load_data('/root/csl/val.tsv')
test_data = load_data('/root/csl/test.tsv')

model_path = '/home/vocust001/pretrained_models/chinese_t5_pegasus_small'
max_len = 512
batch_size = 5
lr = 2e-4

tokenizer = T5PegasusTokenizer.from_pretrained(model_path)


def create_data(data):
    ret = []
    for title, content in data:
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')

        summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
        features = {'input_ids': text_ids, 'decoder_input_ids': summary_ids, 'attention_mask': [1] * len(text_ids),
                    'decoder_attention_mask': [1] * len(summary_ids)}
        ret.append(features)
    return ret


train_data, _ = create_data(train_data)

train_data = KeyDataset(train_data)
train_data = DataLoader(train_data, batch_size=batch_size, collate_fn=default_collate)

model = MT5ForConditionalGeneration.from_pretrained(model_path)

device = 'cuda:1'
model.to(device)
adam = torch.optim.Adam(model.parameters(), lr=lr)


def generate(text, max_length=30):
    max_content_length = max_len - max_length
    feature = tokenizer.encode(text, return_token_type_ids=True, return_tensors='pt',
                               max_length=512)
    feature = {'input_ids': feature}
    feature = {k: v.to(device) for k, v in list(feature.items())}

    gen = model.generate(max_length=max_length, eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id,
                         **feature).cpu().numpy()[0]
    gen = gen[1:]
    gen = tokenizer.decode(gen, skip_special_tokens=True).replace(' ', '')
    return gen


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


best = 0
for _ in range(6):
    model.train()
    for cur in train_data:
        cur = {k: v.to(device) for k, v in cur.items()}
        prob = model(**cur)[0]
        mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
        prob = prob[:, :-1]
        prob = prob.reshape((-1, prob.size(-1)))[mask]
        labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(prob, labels)
        loss.backward()
        adam.step()
        adam.zero_grad()

    # 测试
    model.eval()
    gens = []
    summaries = []
    for (title, content) in valid_data:
        gen = generate(content, max_length=40)
        gens.append(gen)
        summaries.append(title)
    scores = compute_rouges(gens, summaries)
    print(scores)
    rouge_l = scores['rouge-l']
    if rouge_l > best:
        best = rouge_l
        torch.save(model, 'summary_model')
