import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir 
        self.image_dir2 = args.image_dir2 
        self.image_dir_plip = args.image_dir_plip
        self.ann_path = args.ann_path
        self.split_path = args.split_path
        self.h5_dir = args.h5_dir

        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        # ===== 1. 直接读取 split.csv =====
        df = pd.read_csv(self.split_path)
        slide_ids = df[self.split].dropna().tolist()

        print(f"{self.split}: {len(slide_ids)}")

        self.examples = []

        for slide_id in slide_ids:
            slide_id = slide_id.strip()

            # ===== 2. image_dir / plip：用完整 slide_id =====
            image_path = os.path.join(self.image_dir, slide_id + '.pt')
            image_path2 = os.path.join(self.image_dir2, slide_id + '.pt')
            image_plip_path = os.path.join(self.image_dir_plip, slide_id + '.pt')
            h5_path = os.path.join(self.h5_dir, slide_id + '.h5')

            if not os.path.exists(image_path):
                print(f"[MISS IMAGE] {image_path}")
                continue

            if not os.path.exists(image_path2):
                print(f"[MISS PLIP] {image_path2}")
                continue

            if not os.path.exists(image_plip_path):
                print(f"[MISS PLIP] {image_plip_path}")
                continue

            # ===== 3. ann_path：只用前三段 case_id =====
            case_id = '-'.join(slide_id.split('-')[:3])
            anno_path = os.path.join(self.ann_path, case_id, 'annotation')

            if not os.path.exists(anno_path):
                print(f"[MISS ANNO] {anno_path}")
                continue

            with open(anno_path, 'r') as f:
                anno = f.read()

            report_ids = tokenizer(anno)
            report_ids = report_ids[:self.max_seq_length]
            mask = [1] * len(report_ids)

            if len(report_ids) < self.max_seq_length:
                pad_len = self.max_seq_length - len(report_ids)
                report_ids += [0] * pad_len
                mask += [0] * pad_len

            self.examples.append({
                'id': slide_id,
                'image_path': image_path,
                'image_path2': image_path2,
                'image_plip_path': image_plip_path,
                'h5_path': h5_path,   
                'report': anno,
                'ids': report_ids,
                'mask': mask,
                'split': self.split
            })

        print(f"The size of {self.split} dataset: {len(self.examples)}")



    def __len__(self):
        return len(self.examples)

class TcgaImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]

        # ===== 1. load image features =====
        image = torch.load(example['image_path'])[:self.max_fea_length]
        image2 = torch.load(example['image_path2'])[:self.max_fea_length]
        image_plip = torch.load(example['image_plip_path'])[:self.max_fea_length]
        # image = torch.cat((image, image2), dim=1)
        image = torch.cat((image, image_plip), dim=1)

        # ===== 2. load h5 coords =====
        h5_path = example['h5_path']

        try:
            with h5py.File(h5_path, 'r') as f:
                if 'coords' not in f:
                    raise KeyError('coords not found in h5')
                coords = f['coords'][:]
        except Exception as e:
            raise RuntimeError(
                f"[H5 ERROR] failed to load {h5_path}: {e}"
            )

        coords = torch.from_numpy(coords).long()

        # ===== 3. 对齐 patch 数量 =====
        if coords.shape[0] != image.shape[0]:
            n_patches = image.shape[0]
            n_coords = coords.shape[0]

            if n_coords >= n_patches:
                coords = coords[:n_patches]
            else:
                repeat = coords[-1:].repeat(n_patches - n_coords, 1)
                coords = torch.cat([coords, repeat], dim=0)

        # ===== 4. text =====
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = sum(report_masks)

        return (
            example['id'],
            image,
            coords,          # ★ 把 coords 显式返回
            report_ids,
            report_masks,
            seq_length
        )



    def clean_data(self,data):
        cases = {}
        for idx in range(len(data)):
            case_name = data[idx]

            case_id = '-'.join(case_name.split('-')[:3]).split('.')[0]
            cases[case_id] = case_id

        return cases 
    
    def filter_df(self,df, filter_dict):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def df_prep(self,data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

