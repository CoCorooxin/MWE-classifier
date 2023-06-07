from mwe_dataset import MWEDataset
import torch
from collections import Counter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

class MweClassifer(nn.Module):

    def __init__(self, vocab_size, pos_size=None, deprels_size=None, context_size=3, emb_size=64, hidden_size=64,
                 drop_out=0.):

        super(MweClassifer, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, emb_size)
        self.deprel_embedding = nn.Embedding(deprels_size, emb_size)
        self.input_length = 1 + context_size * 2
        self.net = nn.Sequential(
            nn.Linear(emb_size * self.input_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 3),  # output 3 neurons -> 3 classes

            nn.LogSoftmax(dim=1)
        )

    def forward(self, X_tokIDs, X_deprels):
        b, seq = X_tokIDs.shape

        input = (self.word_embedding(X_tokIDs) + self.deprel_embedding(X_deprels))  # Batch, inputsize*emb_size
        # print(input.shape) B, window_size, emb_size
        # input.view(b, -1) B, window_size * emb_size
        return self.net(input.view(b, -1))  # tag_size = 3

    def _init_weights(self):
        pass

    def train_model(self, trainset, epochs=10, lr=1e-3, batch_size=10, device="cpu", reg=None):
        self.to(device)
        # adaptive gradient descent, for every update lr is a function of the amount of change in the parameters
        # optimizer   = torch.optim.Adam()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        loss_fnc = nn.NLLLoss()
        trainloader = trainset.get_loader(batch_size=batch_size)

        train_loss = []

        for i in range(epochs):
            self.train()
            ep_loss = []
            for X, X_deprel, X_pos, y_true in tqdm(trainset.get_loader(batch_size)):
                # print(x.shape)

                optimizer.zero_grad()
                y_hat = self.forward(X, X_deprel)
                # print(y_hat.shape)
                loss_value = loss_fnc(y_hat, y_true)
                ep_loss.append(loss_value.item())
                loss_value.backward()
                optimizer.step()
            loss = sum(ep_loss) / len(ep_loss)
            print(loss)
            train_loss.append(loss)

        print(sum(train_loss) / len(train_loss))

    def validate(self, data_loader, device="cpu"):
        loss_fnc = nn.NLLLoss()
        loss_lst = []
        self.eval()
        pass

    def predict(self, string):
        """

        """

        pass

    def evaluation(self, y_hat, y_gold):
        TP = 0
        FN = 0
        FP = 0
        for logits, gold in zip(y_hat, y_gold):
            pred = int(torch.argmax(logits))
            gold = int(gold)

            if pred * gold != 0:
                TP += 1
            elif gold > 0:
                FN += 1
            elif pred > 0:
                FP += 1
        return TP, FN, FP

        # if torch.argmax(y_hat) != gold and

