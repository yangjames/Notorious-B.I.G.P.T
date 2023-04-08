import random
import os
import pytorch_lightning as pl
from torch.utils.data.dataloader import default_collate
from typing import List, Callable

import torch
from torch import Tensor


class NotoriousDataLoader:
    def __init__(
        self,
        dataset : List[int],
        batch_size : int = 32,
        context_length : int = 8,
        collate_fn : Callable[[int], Tensor] = default_collate,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.collate_fn = collate_fn
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index + 1 >= len(self.dataset):
            raise StopIteration
        remaining = len(self.dataset) - (self.index + 1)
        batch_size = min(remaining, self.batch_size)
        return self.collate_fn([self.get() for _ in range(batch_size)])

    def get(self):
        item = self.dataset[self.index]
        self.index += 1
        return item

    def worker_fn(dataset, index_queue, output_queue):
       while True:
           try:
               index = index_queue.get(timeout=0)
           except queue.Empty:
               continue
           if index is None:
               break
           output_queue.put((index, dataset[index]))


class NotoriousDataset(pl.LightningDataModule):
    def __init__(
        self,
        text_path : str,
        batch_size : int = 32,
        split : float = 0.9,
        context_length : int = 8,
    ):
        super().__init__()
        assert os.path.exists(text_path)
        self.batch_size = batch_size
        self.text_path = text_path
        self.context_length = context_length
        self.split = split
        # self.tokenizer = self._ngram_tokenize
        self.tokenizer = self._word_tokenize
        self.prepare_data()


    def _ngram_tokenize(self, raw_data : str) -> List[str]:
        split_text = [*raw_data]
        vocabulary = sorted(list(set(split_text)))
        return split_text, vocabulary

    def _split_and_cat(self, word : str):
        split_word = []
        cumulative_word = ""
        for char in word:
            if char not in [",", "\n", ".", "?", "!", "(", ")"]:
                cumulative_word += char
            elif len(cumulative_word) == 0:
                split_word += [char]
            else:
                split_word += [cumulative_word, char]
                cumulative_word = ""
        if len(cumulative_word) > 0:
            split_word += [cumulative_word]
        return split_word


    def _word_tokenize(self, raw_data : str) -> List[str]:
        # Split words + punctuation and encode.
        split_text = []
        for i,word in enumerate(self.raw_data.split(" ")):
            split_text += self._split_and_cat(word)
        vocabulary = sorted(list(set(split_text)))
        return split_text, vocabulary
        
    def prepare_data(self):
        # Read in data.
        with open(self.text_path, "r") as f:
            self.raw_data = f.read().lower()
        # Split words + punctuation and encode.
        self._split_text, self._vocabulary = self.tokenizer(self.raw_data)
        self._word_to_int = { word: i for i, word in enumerate(self._vocabulary) }
        self._int_to_word = { i: word for i, word in enumerate(self._vocabulary) }
        self._encoded_text = self._encode(self._split_text)
        # Create inputs and targets.
        num_data_points = len(self._encoded_text) - self.context_length - 1
        dataset = [
            (self._encoded_text[i:i+self.context_length],
             self._encoded_text[i+1:i+1+self.context_length])
            for i in range(num_data_points)
        ]
        random.shuffle(dataset)
        n = int(self.split * len(dataset))
        self._train_split = dataset[:n]
        self._val_split = dataset[n:]

    def _encode(self, vec : List[str]) -> List[int]:
        return torch.Tensor([self._word_to_int[word] for word in vec]).type(torch.long)

    def _decode(self, vec : List[int]) ->List[str]:
        return [self._int_to_word[val] for val in vec]

    def train_dataloader(self):
        train_loader = NotoriousDataLoader(
            self._train_split, batch_size=self.batch_size, context_length=self.context_length)
        return train_loader

    def val_dataloader(self):
        val_loader = NotoriousDataLoader(
            self._val_split, batch_size=self.batch_size, context_length=self.context_length)
        return val_loader



if __name__ == "__main__":
    dataset_path = "../data/notorious_lyrics.txt"
    dataset = NotoriousDataset(dataset_path, batch_size=32, context_length=16, )
    for batch in dataset.train_dataloader():
        assert len(batch) == 2
