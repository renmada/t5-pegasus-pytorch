import os
import argparse
import pytorch_lightning as pl
from utils import *
from t5_copy import T5Copy
import warnings

warnings.filterwarnings('ignore')


class TaskLightModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = T5Copy.from_pretrained(args.model_path)

    def predict_batch(self, batch):
        ids = batch.pop('id')
        pred = self.model.generate(eos_token_id=tokenizer.sep_token_id,
                                   decoder_start_token_id=tokenizer.cls_token_id,
                                   num_beams=3,
                                   input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                   use_cache=True,
                                   max_length=self.args.max_target_length,
                                   src=batch['input_ids']
                                   )
        pred = pred[:, 1:].cpu().numpy()
        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        pred = [s.replace(' ', '') for s in pred]
        return ids, pred

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        res = self.predict_batch(batch)
        return res

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        with open(args.output_path, 'a+') as f:
            for id_, p in zip(*outputs):
                f.write(str(int(id_)) + '\t' + p + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ========================= Train and trainer ==========================
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_start', default=3, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--plugins', type=str, default='ddp_sharded')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=1)
    parser.add_argument('--compute_bleu', action='store_true')
    parser.add_argument('--compute_rouge', action='store_true')

    # ========================= Data ==========================
    parser.add_argument('--train_file', type=str, required=False)
    parser.add_argument('--dev_file', type=str, required=False)
    parser.add_argument('--predict_file', type=str, required=False)
    parser.add_argument('--noise_prob', default=0., type=float)
    parser.add_argument('--max_source_length', default=200, type=int)
    parser.add_argument('--max_target_length', default=150, type=int)
    parser.add_argument('--beams', default=3, type=int)
    parser.add_argument('--num_works', type=int, default=4)

    # ========================= Model ==========================
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str, default='./saved')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    tokenizer = T5PegasusTokenizer.from_pretrained(args.model_path)
    data = EncoderDecoderData(args, tokenizer)
    dataloader = data.get_predict_dataloader()
    trainer = pl.Trainer.from_argparse_args(args, logger=False)
    model = TaskLightModel.load_from_checkpoint(args.resume, args=args)

    trainer.predict(model, dataloader)
