from ast import arg
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from data import Collater, DerivativeDataset, DerivativeLanguage
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from trainer import DerivativeTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle

import os

import yaml


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


# --------- PLEASE FILL THIS IN --------- #

def predict(model, test_pairs, batch_size=2048):
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
    wrongs = 0
    for i, (src, target, prd) in enumerate(
        tqdm(
            zip(src_sentences, target_sentences, prd_sentences),
            desc="scoring",
            total=len(src_sentences),
        )
    ):
        pred_score = score(target, prd)
        total_score += pred_score
        if pred_score==0 and wrongs <= 10 :
            print(f"\n\n\n---- Example {i} ----")
            print(f"src = {src}")
            print(f"target = {target}")
            print(f"prd = {prd}")
            print(f"score = {pred_score}")
            wrongs += 1

    final_score = total_score / len(prd_sentences)
    print(f"{total_score}/{len(prd_sentences)} = {final_score:.4f}")
    return final_score




# ----------------- END ----------------- #




def train_test_split(pairs, train_test_split_ratio):
    random.shuffle(pairs)
    split = int(train_test_split_ratio * len(pairs))
    train_pairs, test_pairs = pairs[0:split], pairs[split:]
    return train_pairs, test_pairs

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

def load_model(path, src_lang, target_lang):    
    model = DerivativeTransformer.load_from_checkpoint(
        os.path.join(path, "model.ckpt"),
        src_lang=src_lang,
        target_lang=target_lang,
    ).to(device)
    return model

def main(args):
    """load, inference, and evaluate"""

    #1. Begin Training Model
    functions, true_derivatives = load_file(args.train_path)    
    
    pairs = list(map(lambda x, y:(x,y), functions, true_derivatives))
    train_pairs, test_pairs = train_test_split( pairs, train_test_split_ratio=0.95)
    
    # Get the Fixed Dictionary That Has Been Collected
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
    
    train_tensors = tokenize(train_pairs, src_lang, target_lang)
    val_tensors = tokenize(test_pairs, src_lang, target_lang)

    train_collate_fn = Collater(src_lang, target_lang)
    train_loader = DataLoader( DerivativeDataset(train_tensors), batch_size=args.batch_size, collate_fn=train_collate_fn, num_workers=8 )
    val_loader = DataLoader(DerivativeDataset(val_tensors), batch_size=args.batch_size, collate_fn=train_collate_fn, num_workers=8 )
    
    model = DerivativeTransformer(src_lang, target_lang, max_len=30, hid_dim=248).to(device)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.path,
        filename="model",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.path,
        callbacks=[checkpoint_callback],
    )
    
    
    trainer.fit(model, train_loader, val_loader)    # train the model
    model = load_model(args.path, src_lang, target_lang) # model
    predict(model, test_pairs, args.batch_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Training")
    parser.add_argument("--train_val_split_ratio", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_path", type=str, default='train.txt')
    parser.add_argument("--test_path", type=str, default='') # no test for grader only
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--enc_heads", type=int, default=8)
    parser.add_argument("--dec_heads", type=int, default=8)
    parser.add_argument("--enc_pf_dim", type=int, default=512)
    parser.add_argument("--dec_pf_dim", type=int, default=512)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--path", type=str, default="best_models")
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True) # make dir 
    
    main(args)
