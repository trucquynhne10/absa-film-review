import sys
import os
import yaml
import torch
import pandas as pd
import numpy as np
import underthesea
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,DistributedSampler
from utils import *
from preprocess import preprocess_fn

# import torch_xla.core.xla_model as xm

def load_dataset_by_filepath(cfg, file_path=None, tokenizer=None):
    if file_path is None or not os.path.exists(file_path):
        print(f"{file_path} not found!")
        sys.exit(1)
    
    # Đọc file CSV
    data = pd.read_csv(file_path)
    data[['character', 'content', 'scene', 'sound']] = data[['character', 'content', 'scene', 'sound']].apply(lambda x: x.str.lower())
    data = data.dropna(subset=['Cmt'])
    # data = txt2df(file_path, cfg['aspect'])
    data['Cmt'] = data['Cmt'].apply(preprocess_fn)

    if tokenizer is not None:
        input_ids = tokenize_inputs(data['Cmt'], tokenizer, num_embeddings=cfg['num_embeddings'])
        attention_masks = create_attn_masks(input_ids)
        
        # Thêm các cột mã hóa vào DataFrame
        data["features"] = input_ids.tolist()
        data["masks"] = attention_masks
    
    return data


def create_dataloader(cfg, data):
	X = data["features"].values.tolist()
	masks = data["masks"].values.tolist()
	label_cols =data.columns.values.tolist()[0:4]

	y = label_encoder_df(data[label_cols].applymap(lambda x: x.lower() if pd.notnull(x) else x)).values.tolist()

	X = torch.tensor(X)

	y = torch.tensor(np.array([onehot_enconder(lb, len(cfg['labels'])) for lb in y]))

	masks = torch.tensor(masks, dtype=torch.long)

	dataset = TensorDataset(X, masks, y)
	sampler = RandomSampler(dataset)
	dataloader = DataLoader(dataset,\
	                        sampler=sampler,\
	                        batch_size=cfg['batch_size'],
	                        num_workers=cfg['num_workers'])
	return dataloader

def create_dataloader_tpu(cfg, data):
	X = data["features"].values.tolist()
	masks = data["masks"].values.tolist()
	label_cols =data.columns.values.tolist()[1:-2]
	y = label_encoder_df(data[label_cols].applymap(lambda x: x.lower() if pd.notnull(x) else x)).values.tolist()

	X = torch.tensor(X)

	y = torch.tensor(np.array([onehot_enconder(lb, len(cfg['labels'])) for lb in y]))

	masks = torch.tensor(masks, dtype=torch.long)

	sampler = DistributedSampler(dataset, shuffle=True)
	dataloader = DataLoader(dataset, sampler=sampler, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

	return dataloader