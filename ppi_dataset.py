import torch
import os
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
import h5py
# from utils.dscript_utils import collate_paired_sequences

class SDNNPPIdataset(Dataset):
    def __init__(self, folder_name, csv_fpath, config=None, fold=-1, is_train=True, out_all=True, esm_pool=False):
        self.aac_path = folder_name
        self.set_path = csv_fpath
        self.folder_name = folder_name
        self.csv_dpath = os.path.dirname(csv_fpath) # 5fold
        self.is_train = is_train # 5fold
        self.k_fold = config['k_fold'] # 5fold
        self.fold = fold # 5fold
        self.data_type = config['data_type']

        self.pair_df = self.__get_pair_df()
        self.out_all = out_all

        self.esm_pool = esm_pool

        
    def __len__(self):
        return len(self.pair_df)
    def __getitem__(self, idx):
        geneA = torch.tensor(list(self.pair_df['aac_A'][idx]))
        geneB = torch.tensor(list(self.pair_df['aac_B'][idx]))
        geneA_2 = torch.tensor(list(self.pair_df['aac_A_2'][idx]))
        label = torch.tensor(self.pair_df['label'][idx])
        label_2 = torch.tensor(self.pair_df['label_2'][idx])
        mute_idx = self.pair_df['mute_idx']
        if '>' in str(mute_idx[idx]):
            mute_id = mute_idx[idx].split('>')[0][:-1]
        else:
            mute_id = -1
        same = self.pair_df['same'][idx]

        # get esm difference
        id_A = self.pair_df['id_A'][idx]
        id_A_2 = self.pair_df['id_A_2'][idx]
        mute_id_int = int(mute_id)
        esm_diff = self.__get_esm_diff(id_A, id_A_2, mute_id_int)

        if self.out_all:
            return geneA.float(), geneB.float(), label, geneA_2.float(), geneB.float(), label_2, same, int(mute_id), self.pair_df['id_A'][idx], self.pair_df['id_B'][idx], esm_diff
        else:
            return geneA.float(), geneB.float(), label, geneA_2.float(), geneB.float(), label_2, same, esm_diff
    
    def __get_pair_df(self):
        if self.k_fold: # 5fold
            if not self.is_train:
                test_fold_fpath = os.path.join(self.csv_dpath, '{}_fold_{}.csv'.format(self.data_type, self.fold))
                print('test_fold_fpath', test_fold_fpath)
                ids = pd.read_csv(test_fold_fpath, sep = ',', names = ['id_A', 'id_B', 'label', 'id_A_2', 'label_2', 'mute_idx'])
            else:
                ids = pd.DataFrame()
                for fold_id in range(self.k_fold):
                    if fold_id == self.fold:
                        continue
                    current_fold_fpath = os.path.join(self.csv_dpath, '{}_fold_{}.csv'.format(self.data_type, fold_id))
                    print('train_fold_fpath', current_fold_fpath)
                    df_i = pd.read_csv(current_fold_fpath, sep = ',', names = ['id_A', 'id_B', 'label', 'id_A_2', 'label_2', 'mute_idx'])
                    ids = pd.concat([ids, df_i], ignore_index=True)
      
        else:
            ids = pd.read_csv(self.set_path, sep = ',', names = ['id_A', 'id_B', 'label', 'id_A_2', 'label_2', 'mute_idx'])
        aac_A = []
        aac_B = []
        aac_A_2 = []

        id_A_list = []
        id_B_list = []
        id_A_2_list = []

        mute_idx = [] 

        with h5py.File(self.aac_path, 'r') as f:
            for i in range(len(ids)):
                try:
                    aac_A.append(list(f[ids['id_A'][i]][:]))
                    aac_B.append(list(f[ids['id_B'][i]][:]))
                    aac_A_2.append(list(f[ids['id_A_2'][i]][:]))
                except:
                    aac_A.append(list(f[ids['id_A'][i].replace(">", "")][:]))
                    aac_B.append(list(f[ids['id_B'][i].replace(">", "")][:]))
                    aac_A_2.append(list(f[ids['id_A_2'][i].replace(">", "")][:]))
                
        mute_idx = ids['mute_idx']


        final_df = pd.DataFrame()
        final_df['id_A'] = ids['id_A']
        final_df['id_B'] = ids['id_B']
        final_df['id_A_2'] = ids['id_A_2']
        final_df['label'] = ids['label']
        final_df['label_2'] = ids['label_2']
        final_df['aac_A'] = aac_A
        final_df['aac_B'] = aac_B
        final_df['aac_A_2'] = aac_A_2
        final_df['mute_idx'] = mute_idx
        final_df['same'] = (ids['id_A'] == ids['id_A_2'])
        return final_df
    
    def __get_embedding_dict(self):
        a = os.listdir(self.embedding_folder_path)
        embed_dict = {}
        for i in a:
            name = i[0:len(i)-3]
            if name not in embed_dict.keys():
                embed_dict[name] = torch.load(self.embedding_folder_path+'/' + i)
        return embed_dict

    def __get_esm_diff(self, id_A, id_A_2, mute_id):
        esm_dpath = 'datasets/embeddings/650M_1'
        id_A_fpath = os.path.join(esm_dpath, '{}.pt'.format(id_A))
        id_A_2_fpath = os.path.join(esm_dpath, '{}.pt'.format(id_A_2))
        if self.esm_pool:
            id_A_muta_feat = torch.load(id_A_fpath)['representations'][33]
            id_A_muta_feat = torch.mean(id_A_muta_feat, dim=0, keepdim=True).squeeze()
            id_A_2_muta_feat = torch.load(id_A_2_fpath)['representations'][33]
            id_A_2_muta_feat = torch.mean(id_A_2_muta_feat, dim=0, keepdim=True).squeeze()
        else:
            id_A_muta_feat = torch.load(id_A_fpath)['representations'][33][mute_id]
            id_A_2_muta_feat = torch.load(id_A_2_fpath)['representations'][33][mute_id]
        esm_diff = torch.abs(id_A_muta_feat - id_A_2_muta_feat)
        return esm_diff

