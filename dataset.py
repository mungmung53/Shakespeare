"""
Created on Thurs May 23, 2024.
@author: Shinhye Lee
"""

import torch
from torch.utils.data import Dataset,Dataloader

""" Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.text = text
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = self.preprocess(text)

    def preprocess(self, text):
        encoded_text = [self.char_to_idx[ch] for ch in text]
        sequences = []
        for i in range(len(encoded_text) - self.seq_length):
            seq = encoded_text[i:i + self.seq_length]
            target = encoded_text[i + 1:i + self.seq_length + 1]
            sequences.append((seq, target))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)

