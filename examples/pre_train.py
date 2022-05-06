import json
from pp_transformers.utils import write_json_lines, read_json_lines
import random
random.seed(128)

file = 'data/conv_summary/DTA-conversation-summary_train.jsonl'

new_lines = []
for line in read_json_lines(file ):
    src = ''
    for item in line['dialogues']:
        src += item['text']
    new_line = {'src': src, 'tgt': ''.join(line['summary'])}


