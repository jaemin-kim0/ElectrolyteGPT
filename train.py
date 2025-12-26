import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--data_name', type=str, default='',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--props', nargs="+", default=['molwt'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)

    args = parser.parse_args()

    set_seed(42)

    wandb.init(project="Electrolyte-GPT", name=args.run_name, mode='offline')

    data = pd.read_csv('datasets/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    if 'Jaemin' in args.data_name:
        train_data = data[data['split'] == 'train'].reset_index(
            drop=True)

    if 'Jaemin' in args.data_name:
        val_data = data[data['split'] == 'test'].reset_index(
            drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    num_props = args.num_props

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
              for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in vsmiles]

    whole_string = ' '.join(smiles + vsmiles)
    whole_string = sorted(list(set(regex.findall(whole_string))))
    print(whole_string)
    
    stoi = { ch:i for i,ch in enumerate(whole_string) }
    with open(f'json/{args.run_name}_stoi.json', 'w') as f:
                json.dump(stoi, f)
    stoi = json.load(open(f'json/{args.run_name}_stoi.json', 'r'))
    itos = { i:ch for ch,i in stoi.items() }


    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0)
    print('train_dataset.vocab_size:', train_dataset.vocab_size)
    print('train_dataset.max_len', train_dataset.max_len)
    print('num_props', num_props)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(mconf)

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=10, ckpt_path=f'pretrained_models/{args.run_name}.pt', block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)
