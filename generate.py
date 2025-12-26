from model import GPT, GPTConfig
from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from get_mol import get_mol
from rdkit.Chem import QED, Crippen, RDConfig
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
        parser.add_argument('--data_name', type=str, default = '', help="name of the dataset to train on", required=False)
        parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 26, help="number of layers", required=False)
        parser.add_argument('--block_size', type=int, default = 54, help="number of layers", required=False)
        parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=False)
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)

        args = parser.parse_args()
        context = "C"

        data = pd.read_csv('/project2/chibueze/jaemink/genMolGPT/MolGPT_DrugDesign/datasets/' + args.data_name + '.csv')
        data = data.dropna(axis=0).reset_index(drop=True)
        data.columns = data.columns.str.lower()

        if 'Jaemin' in args.data_name:
            smiles = data['smiles']

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        lens = [len(regex.findall(i)) for i in smiles]
        max_len = max(lens)
        smiles = [ i + str('<')*(max_len - len(regex.findall(i))) for i in smiles]
        
        jsonfile = args.model_weight.replace('.pt', '')
        stoi = json.load(open('json/' + jsonfile + '_stoi.json', 'r'))
        itos = { i:ch for ch,i in stoi.items() }
        
        print(itos)
        print(len(itos))

        num_props = len(args.props)
        print('num_props', num_props)
        mconf = GPTConfig(args.vocab_size, args.block_size, num_props = num_props,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        model = GPT(mconf)
        model.load_state_dict(torch.load('pretrained_models/' + args.model_weight))
        model.to('cuda')
        print('Model loaded')

        gen_iter = math.ceil(args.gen_size / args.batch_size)
        
        if 'Jaemin' in args.data_name:
            prop2value = {'rascore': [0.99, 0.99],
                        'eox': [4.0, 4.5, 5.0], #newly added
                        'logp_tpsa': [[1.0, 40.0], [1.0, 80.0], [3.0, 40.0], [3.0, 80.0]],
                        'sas_logp': [2.0, 1.0], 
                        'tpsa_sas': [40.0, 2.0], 
                        'sas_molwt': [[3.0, 300]], #newly added
                        'molwt_sas': [[300, 3.0]], #newly added
                        'molwt_rascore': [[100, 0.99], [200, 0.99], [300, 0.99]],
                        'logp_rascore': [[1.0, 0.99], [1.0, 0.99]],
                        'ea': [0.0],
                         'eox_lumo_cond': [4.5, 0.5, -3.0],
                         'eox_cond':[4.5, -3.0],
                         'cond': [-2.0, -1.0, 0.0], #(-3.0, -4.0, -5.0)
                         'ce': [-1.5, -1.7, -2.0], #(-0.3, -0.7, -2.0)
                         'logp_tpsa_sas': [[4.5, 0.5, -1.0]],
                         'eox_cond_ce': [[4.5, -3.0, -3.0]],
                         'cond_ce_eox': [[0.5, -2.1, 6.0]],
                         'condinc': [5.0],
                         'eoxinc': [5.0],
                         'ceinc': [5.0],
                         'condinc_eoxinc_ceinc': [[5.0, 5.0, 5.0]],
                         'condinc_eoxinc_ceinc_lumoinc_visinc_molwtinc_sasinc': [[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]]}
            
        prop_condition = None
        if len(args.props) > 0:
            prop_condition = prop2value['_'.join(args.props)]
        
        all_dfs = []
        all_metrics = []
        count = 0
        
        if prop_condition is None:
            molecules = []
            count += 1
            for i in tqdm(range(gen_iter)):
                    x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                    p = None
                    y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p)
                    for gen_mol in y:
                            completion = ''.join([itos[int(i)] for i in gen_mol])
                            completion = completion.replace('<', '')
                            mol = get_mol(completion)
                            if mol:
                                    molecules.append(mol)

            "Valid molecules % = {}".format(len(molecules))
            mol_dict = []
            for i in molecules:
                    mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})
            results = pd.DataFrame(mol_dict)
            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))
            
            if 'Jaemin' in args.data_name:
                    novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))
            print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
            print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
            print('Novelty ratio: ', np.round(novel_ratio/100, 3))
            results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
            results['unique'] = np.round(len(unique_smiles)/len(results), 3)
            results['novelty'] = np.round(novel_ratio/100, 3)
            all_dfs.append(results)

        else:
            count = 0
            for c in prop_condition:
                molecules = []
                count += 1
                for i in tqdm(range(gen_iter)):
                        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                        p = None
                        if len(args.props) == 1:
                                p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda') # for single condition
                        else:
                                p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to('cuda') # for multiple conditions
                        y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p)
                        for gen_mol in y:
                                completion = ''.join([itos[int(i)] for i in gen_mol])
                                completion = completion.replace('<', '')
                                mol = get_mol(completion)
                                if mol:
                                        molecules.append(mol)

                "Valid molecules % = {}".format(len(molecules))
                mol_dict = []
                for i in molecules:
                        mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})
                results = pd.DataFrame(mol_dict)
                canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                unique_smiles = list(set(canon_smiles))
                
                if 'Jaemin' in args.data_name:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))
                print(f'Condition: {c}')
                print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio/100, 3))
                if len(args.props) == 1:
                        results['condition'] = c
                elif len(args.props) == 2:
                        results['condition'] = str((c[0], c[1]))
                else:
                        results['condition'] = str((c[0], c[1], c[2]))  
                results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
                results['unique'] = np.round(len(unique_smiles)/len(results), 3)
                results['novelty'] = np.round(novel_ratio/100, 3)
                all_dfs.append(results)

        results = pd.concat(all_dfs)
        results.to_csv('datasets/generated_' + args.csv_name + '.csv', index = False)

        unique_smiles = list(set(results['smiles']))
        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        if 'Jaemin' in args.data_name:
                novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))
               

        print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter*count), 3))
        print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio/100, 3))
