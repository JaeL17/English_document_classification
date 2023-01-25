import argparse
import logging
import os
from torch import nn
import pandas as pd
import datetime as dt
import numpy as np
import torch
from os import listdir
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
                
        self.model.cuda()
        self.model_id2label = config.id2label
        self.model_label2id = {_label:_id for _id, _label in config.id2label.items()}
        self.args = args
        self.tokenizer = tokenizer
        self.model = nn.DataParallel(self.model, device_ids=[0,1,2], output_device=0) 
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
    
def pre_process(data):
    data = str(data)
    if type(data) == type('a'):
        data= data.replace('\t', '')
        data = " ".join(data.split())
    return data   

def main(args, input_file, fname2):
    args = AttrDict(args.__dict__)
    logger.info("Training/evaluation parameters {}".format(args))
    
    flag_file_exist = False       
    set_out_id = set()
    print('aaa')

    list_id, list_text, list_gold = [], [], []
    cnt = -1
    with open(input_file, 'r', encoding='utf-8') as fr:
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
    logger.info(f'text number for inference : {total_infer_number}')
        
    list_key = []
    if args.model_type == 'saltluxbert':
        for i in range(1, args.top_n+1, 1):
            list_key.append(f'{i}_pred')
            list_key.append(f'{i}_prob')
    now = dt.datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H:%M")
    output_file = '/workspace/2022_text_classify/patent_infer_data/inference_results/'+fname2+'_'+date_time+'_127.csv'
    #print(output_file)
    fw = open(output_file, 'a', encoding='utf-8')

    classification_helper = ClassificationHelper(args)    
    # list_id. list_gold, list_text, list_result
    for start_idx in range(0, len(list_id), args.buff_size):
        list_temp_id = list_id[start_idx:start_idx+args.buff_size]
        list_temp_gold = list_gold[start_idx:start_idx+args.buff_size]
        list_temp_text = list_text[start_idx:start_idx+args.buff_size]
        logger.info(f'\n[INFERENCE] start_idx : {start_idx}, end_idx : {start_idx + len(list_temp_id)}, all_idx : {total_infer_number}')
        
        if args.model_type == 'saltluxbert':
            list_temp_result = classification_helper.classifyList(list_temp_text, args.top_n)     

        
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
    
    return output_file

def load_infer_data(args):
    add = 1000000
    f_path = '/workspace/2022_text_classify/patent_infer_data/JP_ALL_TAC_202212'
    #file_lst = ['JP_TAC_20221222.csv']
    file_lst = [i for i in listdir('/workspace/2022_text_classify/patent_infer_data/JP_ALL_TAC_202212') if 'csv' in i]
    for fn in file_lst:
        f_path2 = os.path.join(f_path, fn)
        print(f'loading file: {f_path2}')
        f_all = open(f_path2,'rt')
   
        lines_all = f_all.readlines()
        print(f'done loadinf file: {f_path2}')
        print(f'first_time: {lines_all[0]}')
        lines_all2 = [i.split('\t') for i in lines_all]
        div = int(len(lines_all2)/1000000)+1

        print('total_left: ', len(lines_all2))
        print('div: ', div)

        for div_r in range(0, div):
            m_cl, title, abst, numf = [],[],[],[]
            for i in lines_all2[add*div_r:add*(1+div_r)]:
                try:
                    numf_val = i[0]
                    title_val = pre_process(i[1])
                    abst_val = pre_process(i[2])
                    m_cl_val = pre_process(i[3])
                    cn_code = i[4]

                    numf.append([numf_val, '""', cn_code.strip('\n')])
                    m_cl.append(m_cl_val.strip())
                    title.append(title_val.strip())
                    abst.append(abst_val.strip())

                except:
                    pass

            numf2 = ['|'.join(i) for i in numf]
            document =[]
            for a,b,c in zip(title, abst, m_cl):
                doc = a+' [SEP] '+b +' [SEP] '+c
                document.append(doc)
            ids = [i for i in range(len(document))]
            temp_lab = [24329 for i in range(len(document))]

            df = pd.DataFrame()
            df['id'] = ids
            df['document'] = document
            df['label']= temp_lab
            cv_fname = 'infer_tsv/infer_data_' + fn.replace('.csv', '') + '.tsv'
            df.to_csv(cv_fname ,sep = '\t', index=False)      
            inf_file_n = main(args, cv_fname, fn.replace('.csv', ''))

            f_res = open(inf_file_n, 'rt')
            lines_inf = f_res.readlines()
            lines2_inf = [i.split('\t') for i in lines_inf]
            whle = []       
            for i in lines2_inf:
                whle.append([i[2],i[4],i[6],i[3],i[5],i[7]])
            whle_f = ['|'.join(i) for i in whle]
            print('whle_f_len: ', len(whle_f))

            now = dt.datetime.now()
            date_time = now.strftime("%Y_%m_%d_%H:%M")
            fn.replace('.csv', '')
            save_dir ='/workspace/2022_text_classify/patent_infer_data/inference_results4/'+fn.replace('.csv', '')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            f_output_n = f"{save_dir}/{div_r}_{date_time}_{str(len(numf2))}.csv"

            fw = open(f_output_n, 'w', encoding='utf-8')
            fw.write('applno|ltrtno|ntn_cd|class1|class2|class3|score1|score2|score3'+'\n')

            for i, j in zip(numf2,whle_f):
                fw.write(i+'|'+j)
            fw.close()

#             pr_f_n = f"{save_dir}/pr_update.txt"
#             fw2=open(pr_f_n,'a+', encoding='utf-8')
#             fw2.write(str(add*div_r) +' / '+str(add*(1+div_r))+' / '+ str(len(lines_all2))+'\n')
#             fw2.close()
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    #parser.add_argument("--input_file", type=str, default="../../test_data_day1.tsv")
    #parser.add_argument("--output_file", type=str, default="infer_classification_day3.txt")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--decode_seq_len", type=int, default=5)
    
    parser.add_argument("--write_mode", type=int, default=2)
    
    parser.add_argument("--buff_size", type=int, default=4096)
        
    # parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_type", type=str, default="saltluxbert")
    parser.add_argument("--model_name_or_path", type=str, default="/workspace/2022_text_classify/model_results/v_f/patent_bert_large_classification_mgpu/checkpoint-123741")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1024)

    parser.add_argument("--do_lower_case", action='store_true', help="")
    parser.add_argument("--no_cuda", action='store_true', help="")
    
    parser.add_argument("--label_embedding", action='store_true', help="")
    parser.add_argument("--multiclass", action='store_true', help="")    
    parser.add_argument("--do_infer", action='store_true', help="")
    #parser.add_argument("--input_file_path", type=str, required=True)
    args = parser.parse_args()
    #main(args)
    load_infer_data(args)
