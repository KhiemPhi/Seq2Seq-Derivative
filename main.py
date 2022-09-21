import argparse
import os
import pickle
import random
from ast import arg
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import Collater, DerivativeDataset, DerivativeLanguage
from trainer import DerivativeTransformer


MAX_SEQUENCE_LENGTH = 30
TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)

def tokenizer(sentence, lang):
    indexes = [lang.word2index[w] for w in lang.sentence_to_words(sentence)]
    indexes = [lang.SOS_idx] + indexes + [lang.EOS_idx]
    return torch.LongTensor(indexes)


def tokenize(pairs, src_lang, target_lang):
    tensors = [
        (tokenizer(src, src_lang), tokenizer(target, target_lang))
        for src, target in tqdm(pairs, desc="creating tensors")
    ]
    return tensors


def predict(model, test_pairs, batch_size=3000):
    src_sentences, target_sentences = zip(*test_pairs)
        
    pred_tensors = [
            tokenizer(sentence, model.src_lang)
            for sentence in tqdm(src_sentences, desc="creating prediction tensors")
        ]

    collate_fn = Collater(model.src_lang, predict=True)
    pred_dataloader = DataLoader(
        DerivativeDataset(pred_tensors),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    prd_sentences, _, _ = model.predict(pred_dataloader, batch_size=batch_size)  
    
    total_score = 0
    for i, (src, target, prd) in enumerate(
        tqdm(
            zip(src_sentences, target_sentences, prd_sentences),
            desc="scoring",
            total=len(src_sentences),
        )
    ):
        pred_score = score(target, prd)
        total_score += pred_score
        

    final_score = total_score / len(prd_sentences)
    print(f"{total_score}/{len(prd_sentences)} = {final_score:.4f}")
    return final_score

def load_model(path, src_lang, target_lang):   
    model = DerivativeTransformer.load_from_checkpoint(
        path,
        src_lang=src_lang,
        target_lang=target_lang,
    ).to(device)
    return model

def main(args, filepath: str = "test.txt"):
    """load, inference, and evaluate"""  
    functions, true_derivatives = load_file(filepath)
    pairs = list(map(lambda x, y:(x,y), functions, true_derivatives))
    src_lang, target_lang = DerivativeLanguage(), DerivativeLanguage()
    with open('hparams.yaml', 'r') as file: 
        hparams = yaml.safe_load(file)     
        src_lang.word2count = hparams["src_lang"]["word2count"]
        src_lang.word2index = hparams["src_lang"]["word2index"]
        src_lang.index2word = hparams["src_lang"]["index2word"]
        src_lang.n_words =    hparams["src_lang"]["n_words"]
        src_lang.max_length = hparams["src_lang"]["max_length"]
        target_lang.word2count = hparams["target_lang"]["word2count"]
        target_lang.word2index = hparams["target_lang"]["word2index"]
        target_lang.index2word = hparams["target_lang"]["index2word"]
        target_lang.n_words =    hparams["target_lang"]["n_words"]
        target_lang.max_length = hparams["target_lang"]["max_length"]
    model = load_model('model.ckpt', src_lang, target_lang)
    
    predict(model, pairs, args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Training")
    parser.add_argument("--batch_size", type=int, default=3000)
    args = parser.parse_args()
    main(args)
