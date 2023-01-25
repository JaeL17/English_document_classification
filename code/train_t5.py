import argparse
import json
import logging
import os
import shutil
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
from accelerate import Accelerator
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from datahelper import DataHelper
from modelhelper import (
    init_logger,
    set_seed,
    ModelHelper
)

accelerator = Accelerator()
device = accelerator.device

logger = logging.getLogger(__name__)


def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    
    if args.evaluate_test_during_training:
        eval_dataset = test_dataset        
    else:        
        eval_dataset = dev_dataset
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0
    
    list_results_eval = []

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.label_embedding == True:
                inputs["input_label_seq_tensor"] = batch[4]

            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.logging_steps == -1 and args.save_steps == -1:
            if args.evaluate_test_during_training or args.eval_all_checkpoints:
                eval_results = evaluate(args, model, eval_dataset, eval_dataloader, "dev", global_step)
                eval_results['fname'] = "checkpoint-{}".format(global_step)
                list_results_eval.append( eval_results )
            else:
                temp_dict = {}
                temp_dict['fname'] = "checkpoint-{}".format(global_step)
                list_results_eval.append( temp_dict )
            
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(output_dir)
            
#             shutil.copy(os.path.join(args.model_name_or_path, 'vocab.txt'), os.path.join(output_dir, 'vocab.txt'))

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to {}".format(output_dir))

            if args.save_optimizer:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step, list_results_eval


def evaluate(args, model, eval_dataset, eval_dataloader, mode, global_step=None, output_mode="classification"):
    results = {}
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.label_embedding == True:
                inputs["input_label_seq_tensor"] = batch[4]
                # print("[*]", inputs["input_label_seq_tensor"])
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps    
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    # result = compute_metrics(args.task, out_label_ids, preds)
    result = args.eval_func(out_label_ids, preds)
    results.update(result)
    
    fw = open(os.path.join(args.output_dir, "out_{}-{}.txt".format(mode, global_step)), 'w', encoding='utf-8')
    for _label, _pred in zip(out_label_ids, preds):     
        _label = args.dict_id2label.get(str(_label))
        _pred = args.dict_id2label.get(str(_pred))
        BUFF = '{}\t{}\n'.format(_label, _pred)
        fw.write(BUFF)
    fw.close()
    

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results


def main(cli_args):
    # Read from config file and make args
    args = AttrDict(cli_args.__dict__)
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)
    
    init_logger()
    set_seed(args)    

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        
    model_helper = ModelHelper()
    dict_modelset = model_helper.get_modelset(args.model_type)
    
    tokenizer = dict_modelset['tokenizer'].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    ) 
    eval_func = model_helper.get_metrics(args.task)
    args.eval_func = eval_func
    
    # encoder-classifier 구조    
    data_helper = DataHelper(args)
    
    train_dataset = data_helper.get_dataset(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = data_helper.get_dataset(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = data_helper.get_dataset(args, tokenizer, mode="test") if args.test_file else None

    labels = data_helper.get_labels()    
    num_labels = len(labels)
    output_mode = "classification"
    if output_mode == "regression":
        config = dict_modelset['config'].from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels
        )        
        args.num_labels = num_labels
    else:
        config = dict_modelset['config'].from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
        )
        args.num_labels = num_labels
    
    model = dict_modelset['model'].from_pretrained(
        args.model_name_or_path,
        config=config
    )
    model.to(args.device)
    
    args.dict_id2label = {str(k):str(v) for k,v in model.config.id2label.items()}    
    print(args.dict_id2label)

    if args.do_train:
        global_step, tr_loss, list_results_eval = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))
    
    if args.take_lastone:
        fname = list_results_eval[-1]['fname']
        checkpoint = os.path.join(args.output_dir, fname)
        fin_checkpoint = os.path.join(args.output_dir, 'final_model')
        shutil.move(checkpoint, fin_checkpoint)
        
        for e in list_results_eval[:-1]:
            fname = e['fname']
            checkpoint = os.path.join(args.output_dir, fname)
            shutil.rmtree(checkpoint)

    else:
        soretd_list_results_eval = sorted(list_results_eval, key=lambda x:x['f1'], reverse=True)

        fname = soretd_list_results_eval[0]['fname']
        checkpoint = os.path.join(args.output_dir, fname)
        if args.refactor_output:        
            fin_checkpoint = os.path.join(args.output_dir, 'final_model')
            shutil.move(checkpoint, fin_checkpoint)
        else:
            fin_checkpoint = checkpoint

        print(fin_checkpoint)
        model = dict_modelset['model'].from_pretrained(fin_checkpoint)
        model.to(args.device)

        eval_sampler = SequentialSampler(test_dataset)
        eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)    
        test_result = evaluate(args, model, test_dataset, eval_dataloader, mode="test", global_step=global_step)
        test_result['fname'] = fname

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            f_w.write(f"<TEST>\n")
            f_w.write(f"{str(test_result)} \n")
            f_w.write(f"\n<DEV>\n")
            for dict_e in list_results_eval:
                f_w.write(f"{str(dict_e)} \n")

        if args.refactor_output:
            for e in soretd_list_results_eval[1:]:
                fname = e['fname']
                checkpoint = os.path.join(args.output_dir, fname)
                shutil.rmtree(checkpoint)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
   
    cli_parser.add_argument("--task", type=str, default="")
    cli_parser.add_argument("--data_dir", type=str, default="")
    cli_parser.add_argument("--ckpt_dir", type=str, default="")
    cli_parser.add_argument("--train_file", type=str, default="")
    cli_parser.add_argument("--dev_file", type=str, default="")
    cli_parser.add_argument("--test_file", type=str, default="")
    cli_parser.add_argument("--model_type", type=str, default="")
    cli_parser.add_argument("--model_name_or_path", type=str, default="")
    cli_parser.add_argument("--output_dir", type=str, default="")

    cli_parser.add_argument("--max_seq_len", type=int, default=0)
    cli_parser.add_argument("--num_train_epochs", type=int, default=0)    
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=0)
    cli_parser.add_argument("--warmup_proportion", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=0)
    cli_parser.add_argument("--seed", type=int, default=0)
    cli_parser.add_argument("--train_batch_size", type=int, default=0)
    cli_parser.add_argument("--eval_batch_size", type=int, default=0)
    cli_parser.add_argument("--logging_steps", type=int, default=0)
    cli_parser.add_argument("--save_steps", type=int, default=0)
    
    cli_parser.add_argument("--weight_decay", type=float, default=0)
    cli_parser.add_argument("--adam_epsilon", type=float, default=0)
    cli_parser.add_argument("--max_grad_norm", type=float, default=0)
    cli_parser.add_argument("--learning_rate", type=float, default=0)

    cli_parser.add_argument("--evaluate_test_during_training", action='store_true', help="")
    cli_parser.add_argument("--eval_all_checkpoints", action='store_true', help="")
    cli_parser.add_argument("--save_optimizer", action='store_true', help="")
    cli_parser.add_argument("--do_lower_case", action='store_true', help="")
    cli_parser.add_argument("--do_train", action='store_true', help="")
    cli_parser.add_argument("--do_eval", action='store_true', help="")
    cli_parser.add_argument("--no_cuda", action='store_true', help="")
    cli_parser.add_argument("--label_embedding", action='store_true', help="")    
    cli_parser.add_argument("--multiclass", action='store_true', help="")
    cli_parser.add_argument("--do_infer", action='store_true', help="")
    
    cli_parser.add_argument("--do_decoding", action='store_true', help="")
    
    cli_parser.add_argument("--refactor_output", action='store_true', help="")
    cli_parser.add_argument("--take_lastone", action='store_true', help="")
    
    cli_args = cli_parser.parse_args()

    main(cli_args)
