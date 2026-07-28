import torch
from torch.utils.data import Dataset
import random

class ReviewDataset(Dataset):
    def __init__(self, titles, ratings, reviews, tokenizer, max_length=512, dropout=0.2):
        self.titles = list(titles)
        self.ratings = list(ratings)
        self.reviews = list(reviews)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dropout = dropout

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        eos_id = self.tokenizer.special_tokens["<|endoftext|>"]

        prefix = ""
        if random.random() > self.dropout:
            prefix += f"<|startoftitle|>{self.titles[idx]}<|endoftitle|>"
        if random.random() > self.dropout:
            prefix += f"<|rating{self.ratings[idx]}|>"

        prefix_ids = self.tokenizer.encode(prefix)
        body_ids = self.tokenizer.encode(self.reviews[idx]) + [eos_id]
        ids = (prefix_ids + body_ids)[:self.max_length]
        labels = ([-100] * len(prefix_ids) + body_ids)[:self.max_length]

        attn = [1] * len(ids)
        pad_n = self.max_length - len(ids)
        ids    += [eos_id] * pad_n
        labels += [-100]   * pad_n
        attn   += [0]      * pad_n

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
