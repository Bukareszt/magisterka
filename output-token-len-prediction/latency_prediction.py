import datasets
from datasets import load_dataset
import argparse
import transformers
from transformers import AutoConfig, AutoTokenizer, BertModel, DataCollatorWithPadding
from transformers import BertForSequenceClassification
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
import time
from models import BertClassificationModel, BertRegressionModel
from dataloading import generate_dataloaders
from train import train, write_loss_to_file, eval_classification, eval_regression
from evaluate import predict, eval_all_models, plot_model_metrics
from utils import get_output_file_name, get_dataset_path, extract_preview_tokens_from_dataset_path
from logger import Logger

if __name__ == '__main__':
    dataset_name = 'lmsys/lmsys-chat-1m'

    parser = argparse.ArgumentParser()
    parser.add_argument('--all_models', action='store_true', default=False)
    parser.add_argument('--multi_round', action='store_true', default=False)
    parser.add_argument('--head_tail', action='store_true', default=False)
    parser.add_argument('--bert_tiny', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, help='0 for regression, 1 for binary cls, 2 for multi-cls, 3 for multi-cls ordinal, 4 for bi-cls ordinal', default=2)
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands)', default=1000)
    parser.add_argument('--model_name', type=str, help='Name of the LLM to predict for', default='vicuna-13b')
    parser.add_argument('--customized', action='store_true', help='Whether to use customized dataset', default=False)
    parser.add_argument('--dataset_path', type=str, help='Path to customized dataset', default='data/customized_1K')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging', default=True)
    parser.add_argument('--wandb_project', type=str, help='W&B project name', default='latency-prediction')
    parser.add_argument('--log_model', action='store_true', help='Whether to log model checkpoints to W&B', default=False)
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification; 
    # 3: multi-class ordinal classification; 4: bi-class ordinal classification; 

    TASK_TYPE = args.task_type
    FLAG_VICUNA_DATA_ONLY = not args.all_models
    FLAG_FIRST_ROUND_ONLY = not args.multi_round
    FLAG_HEAD_TAIL = args.head_tail

    FLAG_LOAD_MODEL_WEIGHTS = False
    FLAG_SAVE_MODEL_WEIGHTS = True
    if FLAG_LOAD_MODEL_WEIGHTS:
        FLAG_SAVE_MODEL_WEIGHTS = False
    FLAG_BERT_TUNING = True
    FLAG_TINY_BERT = args.bert_tiny
    FLAG_L1_LOSS = args.l1_loss
    FLAG_WRITE_RESULTS = False
    selected_data_size = 1000 * args.data_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_names = ['vicuna-13b', 'wizardlm-13b', 'palm-2', 'llama-2-13b-chat', 'koala-13b',
                   'claude-instant-1', 'oasst-pythia-12b', 'alpaca-13b', 'mpt-7b-chat',
                   'vicuna-7b', 'dolly-v2-12b', 'mpt-30b-chat', 'fastchat-t5-3b', 'chatglm-6b',
                   'claude-1', 'gpt-4', 'vicuna-33b', 'guanaco-33b', 'RWKV-4-Raven-14B',
                   'stablelm-tuned-alpha-7b', 'llama-13b', 'gpt-3.5-turbo', 'llama-2-7b-chat',
                   'claude-2', 'gpt4all-13b-snoozy']
    num_models = len(model_names)

    model_name = 'prajjwal1/bert-tiny' if FLAG_TINY_BERT else 'bert-base-uncased'

    num_classes = 3 if (TASK_TYPE == 1 or TASK_TYPE == 4) else 5
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", legacy=False)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    output_filename = get_output_file_name(
        flag_first_round_only=FLAG_FIRST_ROUND_ONLY,
        flag_vicuna_data_only=FLAG_VICUNA_DATA_ONLY,
        flag_bert_tuning=FLAG_BERT_TUNING,
        flag_tiny_bert=FLAG_TINY_BERT,
        task_type=TASK_TYPE,
        flag_l1_loss=FLAG_L1_LOSS,
        selected_data_size=selected_data_size,
        model_name=args.model_name,
        flag_head_tail=FLAG_HEAD_TAIL
    )
    
    dataset_path = get_dataset_path(
        flag_first_round_only=FLAG_FIRST_ROUND_ONLY,
        flag_vicuna_data_only=FLAG_VICUNA_DATA_ONLY,
        task_type=TASK_TYPE,
        selected_data_size=selected_data_size,
        model_name=args.model_name,
        flag_head_tail=FLAG_HEAD_TAIL,
        customized_path=args.dataset_path if args.customized else None
    )

    num_epochs = 6
    train_batch_size = 16
    test_batch_size = 1
    lr = 1e-5 if FLAG_BERT_TUNING else 1e-4

    # Load dataset with streaming mode for memory efficiency
    print(f'Loading dataset from {dataset_path}...')
    dataset = datasets.load_from_disk(
        dataset_path,
        keep_in_memory=False  # Don't keep whole dataset in memory
    )
    print(f'Dataset loaded: {len(dataset)} samples')
    
    train_dataloader, validation_dataloader, test_dataset, weights = generate_dataloaders(
        dataset, 
        train_batch_size, 
        test_batch_size, 
        bert_tokenizer,
        flag_first_round_only=FLAG_FIRST_ROUND_ONLY,
        task_type=TASK_TYPE,
        num_classes=num_classes
    )
    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size, collate_fn=data_collator)
    config = AutoConfig.from_pretrained(model_name)
    if TASK_TYPE == 1 or TASK_TYPE == 2:
        print('Cross entropy weights: ')
        print(weights)

    # regression or ordinal classification
    if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
        model = BertRegressionModel(config, model_name, hidden_dim=128, 
                                    flag_bert_tuning=FLAG_BERT_TUNING, 
                                    flag_vicuna_data_only=FLAG_VICUNA_DATA_ONLY, 
                                    num_models=num_models).to(device)
        if FLAG_L1_LOSS:
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
    # classification
    elif TASK_TYPE == 1 or TASK_TYPE == 2:
        model = BertClassificationModel(config, model_name, hidden_dim=128, num_classes=num_classes, 
                                        flag_bert_tuning=FLAG_BERT_TUNING, 
                                        flag_vicuna_data_only=FLAG_VICUNA_DATA_ONLY, 
                                        num_models=num_models).to(device)
        # criterion = nn.NLLLoss()
        criterion = nn.NLLLoss(weight=torch.tensor(weights).to(device))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # Initialize the logger if W&B is enabled
    if args.use_wandb:
        # Extract preview tokens from dataset path
        add_response_tokens = extract_preview_tokens_from_dataset_path(dataset_path)
        
        config = {
            'task_type': TASK_TYPE,
            'vicuna_data_only': FLAG_VICUNA_DATA_ONLY,
            'first_round_only': FLAG_FIRST_ROUND_ONLY,
            'head_tail': FLAG_HEAD_TAIL,
            'bert_tuning': FLAG_BERT_TUNING,
            'tiny_bert': FLAG_TINY_BERT,
            'l1_loss': FLAG_L1_LOSS,
            'data_size': selected_data_size,
            'num_epochs': num_epochs,
            'batch_size': train_batch_size,
            'learning_rate': lr,
            'add_response_tokens': add_response_tokens,  # Use the extracted value
            'dataset_path': dataset_path,
        }
        logger = Logger(
            config=config,
            model_name=args.model_name,
            project_name=args.wandb_project,
            enable_logging=True,
            log_model=args.log_model
        )
    else:
        logger = None

    if FLAG_LOAD_MODEL_WEIGHTS:
        model.load_state_dict(torch.load('./models/' + output_filename.split('.')[0] + '.pth'))
        model.to(device)
        print("Loaded model weights from disk.")
    else:
        # Training with logger
        print("Start training...")
        train(model, 
              criterion, 
              optimizer, 
              train_dataloader, 
              validation_dataloader, 
              num_epochs, 
              device,
              flag_bert_tuning=FLAG_BERT_TUNING,
              flag_vicuna_data_only=FLAG_VICUNA_DATA_ONLY,
              task_type=TASK_TYPE,
              flag_write_results=FLAG_WRITE_RESULTS,
              selected_data_size=selected_data_size,
              logger=logger)

    if TASK_TYPE == 0:
        validation_metrics = eval_regression(model, validation_dataloader, device, FLAG_VICUNA_DATA_ONLY)
    elif TASK_TYPE == 3 or TASK_TYPE == 4:
        validation_metrics = eval_regression(model, validation_dataloader, device, FLAG_VICUNA_DATA_ONLY)
        validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device, FLAG_VICUNA_DATA_ONLY)
    else:
        validation_metrics = eval_classification(model, validation_dataloader, device, FLAG_VICUNA_DATA_ONLY)
    print(f'Validation metrics after training:')
    for k, v in validation_metrics.items():
        print(f'{k}: {v:.4f}')

    # Log model checkpoint if W&B is enabled
    if FLAG_SAVE_MODEL_WEIGHTS and logger:
        model_path = './models/' + output_filename.split('.')[0] + '.pth'
        logger.log_model_checkpoint(model, model_path)
    
    # Log final evaluation metrics if W&B is enabled
    if logger and validation_metrics:
        logger.log_metrics(validation_metrics, prefix="final")

    # Inference
    print("Start inference...")
    df = predict(model, test_dataloader, device, 
                FLAG_VICUNA_DATA_ONLY, FLAG_FIRST_ROUND_ONLY, 
                TASK_TYPE, model_names)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/' + output_filename)
    print('Saved results to ./results/' + output_filename)

    if FLAG_VICUNA_DATA_ONLY:
        if TASK_TYPE == 0:
            validation_metrics = eval_regression(model, test_dataloader, device, FLAG_VICUNA_DATA_ONLY)
        elif TASK_TYPE == 3 or TASK_TYPE == 4:
            validation_metrics = eval_regression(model, test_dataloader, device, FLAG_VICUNA_DATA_ONLY)
            validation_metrics = validation_metrics | eval_classification(model, test_dataloader, device, FLAG_VICUNA_DATA_ONLY)
        else:
            validation_metrics = eval_classification(model, test_dataloader, device, FLAG_VICUNA_DATA_ONLY)
        print(f'Metrics on test set:')
        os.makedirs('./metrics', exist_ok=True)
        with open('./metrics/' + output_filename.split('.')[0] + '.txt', 'a') as f:
            for k, v in validation_metrics.items():
                f.write(f'{k}: {v:.4f}\n')
                print(f'{k}: {v:.4f}')
    else:
        if TASK_TYPE == 3 or TASK_TYPE == 4:
            metrics, model_counts = eval_all_models(model, test_dataset, device, 
                                                   model_names, num_classes, TASK_TYPE)
            print(model_counts)
            plot_model_metrics(metrics, model_names, model_counts)

    # Finish the logger
    if logger:
        logger.finish()
