Method Overview:

The best way to model the derivative problem is similar to how one would model the translation from one language to another language problem. 
From this, the derivative problem should be similar to past seq2seq(sequence to sequence) problems. Seq2Seq translation problems achieved
SOA results using the Transformer architecture as described in https://arxiv.org/pdf/1706.03762.pdf (Attention is All You Need). 

I re-implemented a Transformer architecture as described above using a cited Github repo in model.py as inspiration as well as using some other
codes from the same repo. My Transformer model has 4 EncoderLayers and 5 DecoderLayers, with a sentence embedding of hidden dimesion of 248 in 
the vector space. These features of my model resulted in a model of 4.8M parameters, below the 5M threshold as described in the project requirement.

While the derivative problem can be formulated as a seq2seq problem, it differs from past translation models as its sets of tokens are unique.
Mathematics is drastically different from English, which is why I opted to build my own Transformer instead of relying on the BERT or RoBERTa
models provided on HuggingFace. 

Aside from re-implementing a Transformer model, I have also added other optimizations to accuracy and to training:

(1) Novel Tokenizer: The Tokenizer I have created converts word to index that would trained for the Mask-LM task using the re-implemented Transformer. 
The regular expression r"\^|d\w|exp|sin|cos|tan|\d+|\w|\(|\)|\+|-|\*+|" not only captures common math symbols but also captures the derivative symbols
as well as a variety of numbers and not just individual digits. When I did not capture the variety of numbers and just digits, the accuray on the
validation set was 71%. When a variety of numbers was captured, the accuracy is 91%. The dicitonary needed for tokenization is stored in hparams.yaml

(2) PyTorch-Lightning: I wrapped my Transformer module in a PyTorch-Lightning module which allows of learning rate find, batch size finding etc allowing
for faster training. It also eases debugging in later stages. 


Model Results: 

Set                                   Accuracy  
==============================================
Validation-Set-No-Derivative-Sym       69.0%
Validation-Set-Single-Digits           71.0%
Validation-Set( 5% of train.txt)       91.0%
Train-Set (95% of train.txt)           98.0%
All(100% of train.txt)                 96.6%

Code Usage: 

To reproduce training results run: python -u train.py --max_epochs 10 --check_val_every_n_epoch=10

To reproduce testing results run: python -u main.py, test.txt should be default file to be tested with. hparams.yaml contains 
the tokenization dicitonary and model.ckpt is the trained model.

To modifiy batch_size in training: python -u train.py --max_epochs 10 --check_val_every_n_epoch=10 --batch_size [batch_size here]
To modifiy batch_size in testing: python -u main.py --batch_size [batch_size here]

Batch size modification according to available GPU resources

Training Information: 
The model was trained on a single NVIDIA RTX 24GB GPU. It was trained with a batch_size of 3000 for 10 epochs with learning rate 0.0005. 
The model was tested with batch_size of 3000 to fit the entire GPU during one training + testing period. It takes around 10-15 minutes to train 10 epochs.
It takes around 10-15 minutes to evaluate on the whole train set

Important:
Please do not delete hparams.yaml and model.ckpt, they are both needed to reproduce the results cited above. 
