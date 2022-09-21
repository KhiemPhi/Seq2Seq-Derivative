from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from model import Encoder, Decoder
import argparse

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")




class DerivativeTransformer(pl.LightningModule):

    def __init__(self, src_lang, target_lang, max_len=30, hid_dim=256, enc_lays=3, dec_lays=3, enc_heads=8, dec_heads=8, enc_pf_dim=512, dec_pf_dim=512, enc_dropout=0.1, dec_dropout=0.1, lr=0.0005):
        super().__init__()


        self.save_hyperparameters()
        del self.hparams["src_lang"]
        del self.hparams["target_lang"]

        self.src_lang = src_lang
        self.target_lang = target_lang

        self.encoder = Encoder(src_lang.n_words, hid_dim, enc_lays, enc_heads, enc_pf_dim, enc_dropout, device)
        self.decoder = Decoder(target_lang.n_words, hid_dim, dec_lays, dec_heads, dec_pf_dim, dec_dropout, device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.target_lang.PAD_idx)
        self.max_len = max_len
        self.to(device)
        
    
    def initialize_weights(self):
        """
            This function inits weight of models uniformly for better training
        """
        def _initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.encoder.apply(_initialize_weights)
        self.decoder.apply(_initialize_weights)
    
    def mask_src(self, src): 

        src_mask = (src != self.src_lang.PAD_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
    
    def mask_target(self, target):

        target_pad_mask = self.mask_src(target)
        target_len = target.shape[1]
        target_sub_mask = torch.tril(torch.ones((target_len, target_len)).type_as(target)).bool()

        target_mask = target_pad_mask & target_sub_mask

        return target_mask
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def predict(self, pred_dataloader, batch_size=128):
        """Efficiently predict a list of sentences"""
        

        sentences = []
        words = []
        attention = []
        for batch in tqdm(pred_dataloader, desc="predict batch num"):
            preds = self.predict_batch(batch.to(device))
            pred_sentences, pred_words, pred_attention = preds
            sentences.extend(pred_sentences)
            words.extend(pred_words)
            attention.extend(pred_attention)

        # sentences = [num pred sentences]
        # words = [num pred sentences, trg len]
        # attention = [num pred sentences, n heads, trg len, src len]

        return sentences, words, attention

    
    def predict_batch(self, batch):
        """Predicts on a batch of src_tensors."""
        # batch = src_tensor when predicting = [batch_size, src len]

        src_tensor = batch
        src_mask = self.mask_src(batch)

        # src_mask = [batch size, 1, 1, src len]

        enc_src = self.encoder(src_tensor, src_mask)

        # enc_src = [batch size, src len, hid dim]

        target_indexes = [[self.target_lang.SOS_idx] for _ in range(len(batch))]

        # target_indexes = [batch_size, cur target len = 1]

        target_tensor = torch.LongTensor(target_indexes).to(self.device)

        # target_tensor = [batch_size, cur target len = 1]
        # cur target len increases during the for loop up to the max len

        for _ in range(self.max_len):

            target_mask = self.mask_target(target_tensor)

            # target_mask = [batch size, 1, cur target len, cur target len]

            output, attention = self.decoder(target_tensor, enc_src, target_mask, src_mask)

            # output = [batch size, cur target len, output dim]

            preds = output.argmax(2)[:, -1].reshape(-1, 1)

            # preds = [batch_size, 1]

            target_tensor = torch.cat((target_tensor, preds), dim=-1)

            # target_tensor = [batch_size, cur target len], cur target len increased by 1

        src_tensor = src_tensor.detach().cpu().numpy()
        target_tensor = target_tensor.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()

        pred_words = []
        pred_sentences = []
        pred_attention = []
        for src_indexes, target_indexes, attn in zip(src_tensor, target_tensor, attention):
            
            src_eosi = np.where(src_indexes == self.src_lang.EOS_idx)[0][0]
            _target_eosi_arr = np.where(target_indexes == self.target_lang.EOS_idx)[0]
            if len(_target_eosi_arr) > 0:  # check that an eos token exists in target
                target_eosi = _target_eosi_arr[0]
            else:
                target_eosi = len(target_indexes)

           
            target_indexes = target_indexes[1:target_eosi]

            
            attn = attn[:, :target_eosi, :src_eosi]

            words = [self.target_lang.index2word[index] for index in target_indexes]
            sentence = self.target_lang.words_to_sentence(words)
            pred_words.append(words)
            pred_sentences.append(sentence)
            pred_attention.append(attn)

        

        return pred_sentences, pred_words, pred_attention

    def forward(self, src, target):
        
        src_mask = self.mask_src(src)
        target_mask = self.mask_target(target)

        enc_out = self.encoder(src, src_mask)
        dec_out, attention = self.decoder(target, enc_out, target_mask, src_mask)

        return dec_out, attention

    def get_loss(self, batch): 
        src, target = batch 
        src = src.to(device)
        target = target.to(device)
        output, _ = self(src, target[:, :-1])       
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = target[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, target)
        return loss

    def training_step(self, batch, batch_idx):        
        loss = self.get_loss(batch)
        self.log("train_loss", loss)        
        return loss

    def validation_step(self, batch, batch_idx):        
        loss = self.get_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)



