import jieba
import numpy as np
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer
from tokenizer import T5PegasusTokenizer

config_path = 'D:\\BaiduNetdiskDownload\\chinese_t5_pegasus_base\\chinese_t5_pegasus_base\\config.json'
checkpoint_path = 'D:\\BaiduNetdiskDownload\\chinese_t5_pegasus_base\\chinese_t5_pegasus_base\\model.ckpt'
dict_path = 'D:\\BaiduNetdiskDownload\\chinese_t5_pegasus_base\\chinese_t5_pegasus_base\\vocab.txt'
torch_model = './'

if __name__ == "__main__":
    text = '蓝蓝的天上有一朵白白的云'

    # torch版本
    tokenizer = T5PegasusTokenizer.from_pretrained(torch_model)
    ids = tokenizer.encode(text, return_tensors='pt')
    model = MT5ForConditionalGeneration.from_pretrained(torch_model)
    output = model.generate(ids,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.sep_token_id,
                            top_k=1,
                            max_length=30).numpy()[0]
    torch_res = ''.join(tokenizer.decode(output[1:])).replace(' ', '')

    # bert4keras版本
    max_c_len = 256
    max_t_len = 29
    tokenizer = Tokenizer(
        dict_path,
        do_lower_case=True,
        pre_tokenize=lambda s: jieba.cut(s, HMM=False)
    )
    t5 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='t5.1.1',
        return_keras_model=False,
        name='T5',
    )

    encoder = t5.encoder
    decoder = t5.decoder
    model = t5.model


    class AutoTitle(AutoRegressiveDecoder):
        """seq2seq解码器
        """

        @AutoRegressiveDecoder.wraps(default_rtype='probas')
        def predict(self, inputs, output_ids, states):
            c_encoded = inputs[0]
            return self.last_token(decoder).predict([c_encoded, output_ids])

        def generate(self, text, topk=1):
            c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
            c_encoded = encoder.predict(np.array([c_token_ids]))[0]
            output_ids = self.beam_search([c_encoded], topk=topk)  # 基于beam search
            return tokenizer.decode(output_ids)


    autotitle = AutoTitle(
        start_id=tokenizer._token_start_id,
        end_id=tokenizer._token_end_id,
        maxlen=max_t_len
    )

    print('原文', text)
    print('bert4keras预测' + '\t' + autotitle.generate(text))
    print('torch预测     ' + '\t' + ''.join(tokenizer.decode(output[1:])))
