import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log

"""
I. The first vanilla model implements a MLP with 2 hidden layers; the input is fixed length 
with window size as hyperparameter
"""
class MLP_baseline(nn.Module):
    """
    A vanilla MLP with 2 hidden layers
    """
    def __init__(self, emb_size, hidden_size, drop_out = 0.):

        super(MLP_baseline, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.Dropout(drop_out),
            nn.ReLU()
        )

    def forward(self, embeddings):

        return self.net(embeddings)

"""
The following models all have a CRF encode and decoder
"""
"""
II. The Second model implemented is a one layer bi-directional RNN sequential model
It is said to be benefitial to our sequential task
"""


"""
III. The third model implemented is a bi-directional RNN with an attention layer at the output, I figure
it would be benefitial if the RNN can look back at the sequence of prediction and choose which tag
in the sequence to attend to when predicting the current token
"""


class Attention(nn.Module):
    def __init__(self, emb_size, drop_out=0.):
        super(Attention, self).__init__()
        self.hidden_size = emb_size
        self.Wq = nn.Linear(emb_size, emb_size)  # query
        self.Wk = nn.Linear(emb_size, emb_size)  # key
        self.Wv = nn.Linear(emb_size, emb_size)  # contextual value
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        self.FFW= nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(emb_size*4, emb_size),
            nn.ReLU()
        )

    def forward(self, xembeddings):
        bs, seq_len, emb_size = xembeddings.shape

        #positional_embeds = self.positional_encoding(xembeddings)

        # Add positional embeddings to input embeddings
        #xembeddings = xembeddings + positional_embeds

        K = self.Wk(xembeddings)
        Q = self.Wq(xembeddings)
        V = self.Wv(xembeddings) #bs, seq, embsize
        scores = Q @ K.transpose(-2, -1) #bs, seq, seq

        attention_wights = F.softmax(scores / sqrt(emb_size), dim=-1)  # bs, seq, seq
        V = V + attention_wights @ (self.ln1(V))
        V = V + self.FFW(self.ln2(V)) #  bs, seq, seq @ bs, seq, embsize = bs, seq, embsize
        #print(out.shape)
        return V   #+ self.FFW(self.ln2(attention_wights @ (self.ln1(V))))

class AttentionRNN(nn.Module):
    def __init__(self,emb_size, hidden_size, drop_out=0.):
        super(AttentionRNN, self).__init__()
        self.rnn = nn.RNN(emb_size, hidden_size // 2, batch_first=True, num_layers=1, bidirectional=True)
        self.attention = Attention(emb_size, drop_out= drop_out)  # Add attention layer

    def forward(self, xembeddings):
        logits, _ = self.rnn(xembeddings)
        attn_output = self.attention(logits)
        return attn_output, _

    def __call__(self, xembeddings):
        return self.forward(xembeddings)
"""
IV a LSTM model is also implemented.this one is just to see if the RNN with Attention achieve better result than the classic LSTM implementation
"""
