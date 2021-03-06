# t5-pegasus pytorch
追一科技开源的[t5-pegasus](https://github.com/ZhuiyiTechnology/t5-pegasus)的pytorch版本
## 下载

- base版本  百度网盘: https://pan.baidu.com/s/1TGthgU22iZp_y1MZZMs1YA 提取码: j15c
- samll版本 百度网盘：https://pan.baidu.com/s/1Nw5wLb6KmcCOHSLtKssq8w 提取码：9rd8
## how to use
pytorch1.7.0 + transformers4.3.3

```python
from tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

model_path = './'
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
text = '蓝蓝的天上有一朵白白的云'
ids = tokenizer.encode(text, return_tensors='pt')
output = model.generate(ids,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.sep_token_id,
                            max_length=30).numpy()[0]
print(''.join(tokenizer.decode(output[1:])).replace(' ', ''))
```
### huggingface model hub

|  模型名	   | MODEL_NAME  |
|  ----  | ----  |
| t5-pegasus-base  | imxly/t5-pegasus |
| t5-pegasus-small  | imxly/t5-pegasus-small |

## 对比bert4keras的原版
```
python test.py
```
输出
```python
原文:蓝蓝的天上有一朵白白的云
bert4keras预测	《蓝蓝的天上有一朵白白的云》是蓝蓝的天上有一朵白白的云创作的网络小说，发表于
torch预测     	《蓝蓝的天上有一朵白白的云》是蓝蓝的天上有一朵白白的云创作的网络小说，发表于
```