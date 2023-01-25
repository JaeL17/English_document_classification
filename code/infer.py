import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from fastprogress.fastprogress import progress_bar
from attrdict import AttrDict

from modelhelper import (
    init_logger,
    ModelHelper
)

init_logger()
logger = logging.getLogger(__name__)


'''

- 문장 리스트가 주어지면, 모델에서 추론하는 기능
- 문장 리스트별로 캐싱
- 출력 파일과 비교해서 중복체크 및 없는 것 새롭게 추론
- flat_model 지원

'''

class ClassificationHelper():
    def __init__(self, args):
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

        model_helper = ModelHelper()
        dict_modelset = model_helper.get_modelset(args.model_type)

        tokenizer = dict_modelset['tokenizer'].from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )        
        config = dict_modelset['config'].from_pretrained(args.model_name_or_path)   
        # self.model = dict_modelset['model'].from_pretrained(args.model_name_or_path)
        
        self.model = dict_modelset['model'](config)
        # print(self.model)
        self.model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin")))
                
        self.model.to(args.device)
        self.model_id2label = config.id2label
        self.model_label2id = {_label:_id for _id, _label in config.id2label.items()}
        self.args = args
        self.tokenizer = tokenizer
        
        # model, tokenizer, args

    
    def classifyList(self, list_text, top_n=1):
        list_result = []
        
        batch_encoding = self.tokenizer.batch_encode_plus(
            [(text_data, None) for text_data in list_text],
            max_length=args.max_seq_len,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )
        
        all_input_ids = torch.tensor([e for e in batch_encoding['input_ids']], dtype=torch.long)
        all_attention_mask = torch.tensor([e for e in batch_encoding['attention_mask']], dtype=torch.long)
        all_token_type_ids = torch.tensor([e for e in batch_encoding['token_type_ids']], dtype=torch.long)

        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        sequence_sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sequence_sampler, batch_size=self.args.eval_batch_size)

        for batch in progress_bar(dataloader):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": None}
                outputs = self.model(**inputs)
                
                logits = outputs.logits
                logits = torch.nn.functional.softmax(logits, dim=1)
                
                np_logits = logits.detach().cpu().numpy()
                np_preds = np.argsort(np_logits, axis=1)
                
                topn_pred = np_preds[:,-top_n:]
                topn_prob = np.take_along_axis(np_logits, topn_pred, axis=1)
                
                for list_pred, list_prob in zip(topn_pred, topn_prob):
                    dict_result = {}
                    for cnt, (_pred, _prob) in enumerate(zip(list_pred, list_prob)):
                        _pred = self.model_id2label.get(_pred)
                        topn_idx = top_n-cnt
                        dict_result.update({f'{topn_idx}_pred':str(_pred), f'{topn_idx}_prob':str(_prob)})
                    list_result.append(dict_result)

        return list_result  # [{'1_pred':'', '1_prob':'', '2_pred':'', '2_prob':''}, ...]
    
    
    def classifyList_decoding(self, list_text, top_n=1, decode_seq_len=5):
        list_result = []
        
        batch_encoding = self.tokenizer.batch_encode_plus(
            [(text_data, None) for text_data in list_text],
            max_length=args.max_seq_len,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )
        
        all_input_ids = torch.tensor([e for e in batch_encoding['input_ids']], dtype=torch.long)
        all_attention_mask = torch.tensor([e for e in batch_encoding['attention_mask']], dtype=torch.long)     
        all_token_type_ids = torch.tensor([e for e in batch_encoding['token_type_ids']], dtype=torch.long)
        all_decoder_inputs = torch.tensor([self.model_label2id.get('#') for _ in range(len(batch_encoding['input_ids']))], dtype=torch.long)

        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_decoder_inputs, all_token_type_ids)
        sequence_sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sequence_sampler, batch_size=self.args.eval_batch_size)

        for batch in progress_bar(dataloader):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            
            list_logits, list_preds = [], []
            with torch.no_grad():
                inputs = {
                    "encoder_input_ids": batch[0],
                    "attention_mask": batch[1],
                    "decoder_input_ids": batch[2].view(-1, 1),
                    "token_type_ids": batch[3],
                }
                
                for i in range(decode_seq_len):
                    outputs = self.model(**inputs)
                    logits = outputs['logits']

                    temp_logits = torch.nn.functional.softmax(logits, dim=2) # [batch_size, decoder_seqlen, num_labels]
                    temp_preds = torch.argmax(logits, dim=2) # [batch_size, decoder_seqlen]

                    list_logits.append(temp_logits[:,-1, :])
                    list_preds.append(torch.unsqueeze(temp_preds[:,-1], 1))

                    inputs["decoder_input_ids"] = torch.cat((inputs["decoder_input_ids"], temp_preds[:,-1].view(-1, 1)), dim=1)
                
            total_logits = torch.cat(list_logits, dim=1)
            total_pred = torch.cat(list_preds, dim=1)
            
            np_logits = total_logits.detach().cpu().numpy().reshape(-1, decode_seq_len, len(self.model_id2label))
            np_preds = np.argsort(np_logits, axis=2)
            
            topn_pred = np_preds[:,:,-top_n:]
            topn_prob = np.take_along_axis(np_logits, topn_pred, axis=2)

            for list_pred, list_prob in zip(topn_pred, topn_prob):
                dict_result = {}
                for decode_idx in range(decode_seq_len):
                    list_pred_ele = list_pred[decode_idx]
                    list_prob_ele = list_prob[decode_idx]
                    for cnt, (_pred, _prob) in enumerate(zip(list_pred_ele, list_prob_ele)):
                        _pred = self.model_id2label.get(int(_pred))
                        topn_idx = top_n-cnt
                        dict_result.update({f'{decode_idx+1}_{topn_idx}_pred':str(_pred), f'{decode_idx+1}_{topn_idx}_prob':str(_prob)})
                list_result.append(dict_result)

        return list_result  # [{'1_pred':'', '1_prob':'', '2_pred':'', '2_prob':''}, ...]


def main(args):
    args = AttrDict(args.__dict__)
    logger.info("Training/evaluation parameters {}".format(args))
    
    flag_file_exist = False       
    set_out_id = set()
    if os.path.exists(args.output_file):
        flag_file_exist = True
        with open(args.output_file, 'r', encoding='utf-8') as fr:        
            for line in fr.readlines():
                _id = line.split('\t')[0]
                set_out_id.add(_id)

    list_id, list_text, list_gold = [], [], []
    cnt = -1
    with open(args.input_file, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            cnt += 1
            if cnt == 0:
                logger.info(f'First line of input {line}')
                continue
            try:
                sp_line = line.strip().split('\t')
                if sp_line[0] in set_out_id:
                    continue
                if len(sp_line) == 3:
                    list_id.append(sp_line[0])
                    list_text.append(sp_line[1])
                    list_gold.append(sp_line[2])
                elif len(sp_line) == 2:
                    list_id.append(sp_line[0])
                    list_text.append(sp_line[1])
                    list_gold.append('')
            except:
                print('[ERROR]', sp_line)
                pass
    total_infer_number = len(list_text)
    logger.info(f'text number for infrence : {total_infer_number}')
        
    list_key = []
    if args.model_type == 'koelectra-base-v3':
        for i in range(1, args.top_n+1, 1):
            list_key.append(f'{i}_pred')
            list_key.append(f'{i}_prob')
    elif args.model_type == 'patent_electra_exp3_3' :
        for i in range(1, args.decode_seq_len+1, 1):
            for j in range(1, args.top_n+1, 1):
                list_key.append(f'{i}_{j}_pred')
                list_key.append(f'{i}_{j}_prob')
    
    fw = open(args.output_file, 'a', encoding='utf-8')
    if flag_file_exist is False:
        if args.write_mode == 1:
            BUFF = '\t'.join(['id', 'gold', 'text'] + list_key)
        if args.write_mode == 2:
            BUFF = '\t'.join(['id', 'gold'] + list_key)        
        fw.write(BUFF + '\n')

    classification_helper = ClassificationHelper(args)    
    # list_id. list_gold, list_text, list_result
    for start_idx in range(0, len(list_id), args.buff_size):
        list_temp_id = list_id[start_idx:start_idx+args.buff_size]
        list_temp_gold = list_gold[start_idx:start_idx+args.buff_size]
        list_temp_text = list_text[start_idx:start_idx+args.buff_size]
        logger.info(f'\n[INFERENCE] start_idx : {start_idx}, end_idx : {start_idx + len(list_temp_id)}, all_idx : {total_infer_number}')
        
        if args.model_type == 'koelectra-base-v3':
            list_temp_result = classification_helper.classifyList(list_temp_text, args.top_n)        
        elif args.model_type == 'patent_electra_exp3_3' :
            list_temp_result = classification_helper.classifyList_decoding(list_temp_text, args.top_n)
        
        for i in range(len(list_temp_id)):
            result_BUFF = '\t'.join([list_temp_result[i][key] for key in list_key])
            if args.write_mode == 1:
                BUFF = '\t'.join([str(e) for e in [list_temp_id[i], list_temp_gold[i], list_temp_text[i], result_BUFF]])
            elif args.write_mode == 2:
                # print('[***]', list_temp_id[i], list_temp_gold[i], result_BUFF)
                BUFF = '\t'.join([str(e) for e in [list_temp_id[i], list_temp_gold[i], result_BUFF]])
            fw.write(BUFF + '\n')
        fw.flush()
    fw.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--decode_seq_len", type=int, default=5)
    
    parser.add_argument("--write_mode", type=int, default=2)
    
    parser.add_argument("--buff_size", type=int, default=1024)
        
    # parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    
    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=0)

    parser.add_argument("--do_lower_case", action='store_true', help="")
    parser.add_argument("--no_cuda", action='store_true', help="")
    
    parser.add_argument("--label_embedding", action='store_true', help="")
    parser.add_argument("--multiclass", action='store_true', help="")    
    parser.add_argument("--do_infer", action='store_true', help="")

    args = parser.parse_args()
    main(args)

