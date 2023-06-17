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
II a bi LSTM model is also implemented.this one is just to see if the classic LSTM implementation achieve the benchmark
"""
class LSTMmwe(nn.Module):
    def __init__(self, embsize, hidden_size, device):
        super(LSTMmwe, self).__init__()
        self.lstm = nn.LSTM(embsize, hidden_size // 2, batch_first=True, num_layers=1, bidirectional=True, device = device)


    def forward(self, xembeddings, mask):
        # x: input sequences of shape (B, S, E)
        # mask: mask tensor of shape (B, S)

        # Apply mask to the embeds sequences
        masked_x = xembeddings * (mask.unsqueeze(-1))

        # Pass the masked input sequences to the RNN layer
        output, _ = self.lstm(masked_x)

        return output, _

"""
out of scope models
-------------------------------
"""

"""
IV. The Second model implemented is a bi-directional Lstm with an attention layer at the output, just a try
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

    def forward(self, xembeddings, mask):
        bs, seq_len, emb_size = xembeddings.shape

        #positional_embeds = self.positional_encoding(xembeddings)

        # Add positional embeddings to input embeddings
        #xembeddings = xembeddings + positional_embeds

        K = self.Wk(xembeddings)
        Q = self.Wq(xembeddings)
        V = self.Wv(xembeddings) #bs, seq, embsize
        scores = Q @ K.transpose(-2, -1) #bs, seq, seq
        scores = scores.masked_fill((~mask).unsqueeze(1), float('-inf'))
        attention_wights = F.softmax(scores / sqrt(emb_size), dim=-1)  # bs, seq, seq
        #with residual connections
        #V = V + attention_wights @ (self.ln1(V))
        #V = V + self.FFW(self.ln2(V)) #  bs, seq, seq @ bs, seq, embsize = bs, seq, embsize
        #print(out.shape)
        return V + self.FFW(self.ln2(attention_wights @ self.ln1(V)))

class AttentionLSTM(nn.Module):
    def __init__(self,emb_size, hidden_size, drop_out=0., device = "cpu"):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(emb_size, hidden_size // 2, batch_first=True, num_layers=1, bidirectional=True, device = device)
        self.attention = Attention(hidden_size, drop_out= drop_out)  # Add attention layer

    def forward(self, xembeddings, mask):
        logits, _ = self.lstm(xembeddings)
        attn_output = self.attention(logits, mask)
        return attn_output, _

    def __call__(self, xembeddings, mask):
        return self.forward(xembeddings, mask)

