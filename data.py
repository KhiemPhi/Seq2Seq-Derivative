import re 
import random 

from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DerivativeLanguage: 
    PAD_idx = 0
    SOS_idx = 1
    EOS_idx = 2
    UNK_idx = 3

    def __init__(self):
        self.word2count = {}
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = 4
        self.max_length = 30
    
    def sentence_to_words(self, sentence):
        # de words, exp, sin, cos, tan, ^ symbols are added to vocab
        return re.findall(r"\^|d\w|exp|sin|cos|tan|\d+|\w|\(|\)|\+|-|\*+|", sentence.strip().lower())

    def words_to_sentence(self, words):
        return "".join(words)
    
    def add_sentence(self, sentence):
        words = self.sentence_to_words(sentence)

        if len(words) > self.max_length:
            self.max_length = len(words)

        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    @classmethod
    def create_vocabs(cls, pairs, src_kwargs={}, target_kwargs={}):

        src_lang = cls(**src_kwargs)
        target_lang = cls(**target_kwargs)

        for src, target in tqdm(pairs, desc="creating vocabs"):
            src_lang.add_sentence(src)
            target_lang.add_sentence(target)

        return src_lang, target_lang
    

class Collater:
    def __init__(self, src_lang, target_lang=None, predict=False):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.predict = predict

    def __call__(self, batch):
        
        if self.predict:            
            return nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=self.src_lang.PAD_idx
            )

        src_tensors, target_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.src_lang.PAD_idx
        )
        target_tensors = nn.utils.rnn.pad_sequence(
            target_tensors, batch_first=True, padding_value=self.target_lang.PAD_idx
        )
        return src_tensors, target_tensors


class DerivativeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)