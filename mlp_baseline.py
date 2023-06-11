import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log
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
        b, seq, embsize= embeddings.shape
        return self.net(embeddings.view(b, -1))


class AttentionMLP(nn.Module):
    def __init__(self, emb_size, drop_out=0.):

        super(AttentionMLP, self).__init__()
        self.hidden_size = emb_size
        self.Wq = nn.Linear(emb_size, emb_size)  # query
        self.Wk = nn.Linear(emb_size, emb_size)  # key
        self.Wv = nn.Linear(emb_size, emb_size)  # contextual value
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        self.FFW= nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.Dropout(drop_out),
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
        return V
