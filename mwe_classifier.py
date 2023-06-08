from mwe_dataset import MWEDataset
import torch
from collections import Counter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec


class MweClassifer(nn.Module):

    def __init__(self, toks_vocab, tags_vocab, window_size=0, emb_size=64, hidden_size=64, pretrainedw2v=None,
                 drop_out=0.):

        super(MweClassifer, self).__init__()

        self.word_embedding = nn.Embedding(len(toks_vocab), emb_size)
        self.window_size = window_size
        self.input_length = 1 + window_size * 2
        self.toks_vocab = toks_vocab
        self.tags_vocab = tags_vocab

        self.net = nn.Sequential(
            nn.Linear(emb_size * self.input_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, len(tags_vocab)),  # output # of classes
            nn.LogSoftmax(dim=1)
        )

        if pretrainedw2v:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.pretrainedw2v_loader(pretrainedw2v).wv.vectors))

    @staticmethod
    def pretrainedw2v_loader(self, path_to_pretrained=None):
        if not path_to_pretrained:
            # Download the pretrained French Word2Vec model https://fauconnier.github.io/#data
            model_name = 'frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin.gz'
            model = api.load(model_name)
        else:
            model = KeyedVectors.load_word2vec_format(path_to_pretrained, binary=True)
        return model

    def forward(self, Xtoks_IDs):
        b, seq = Xtoks_IDs.shape

        input = self.word_embedding(Xtoks_IDs)  # Batch, inputsize*emb_size
        # print(input.shape) #B, window_size, emb_size
        # input.view(b, -1) B, window_size * emb_size
        return self.net(input.view(b, -1))  # tag_size = 3

    def _init_weights(self):
        pass

    def train_model(self, trainset, epochs=10, lr=1e-3, batch_size=10, device="cpu", reg=None):
        """
        the train data is in form of nested lists: [sentences[tokens]]
        """
        self.to(device)
        # adaptive gradient descent, for every update lr is a function of the amount of change in the parameters
        # optimizer   = torch.optim.Adam()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        loss_fnc = nn.NLLLoss()
        trainloader = trainset.get_loader(batch_size=batch_size, shuffle=True)

        train_loss = []

        for e in range(epochs):
            self.train()

            ep_loss = []

            for X_toks, Y_gold in tqdm(trainloader):
                # print(x.shape)
                optimizer.zero_grad()
                y_hat = self.forward(X_toks)

                # print(y_hat.shape)
                loss_value = loss_fnc(y_hat, Y_gold)
                ep_loss.append(loss_value.item())
                loss_value.backward()
                optimizer.step()
            loss = sum(ep_loss) / len(ep_loss)
            print(loss)
            train_loss.append(loss)

            # print("Epoch %d | Mean train loss  %.4f | Mean dev loss %.4f"%(e,loss, devloss) )
            print("Epoch %d | Mean train loss  %.4f" % (e, loss))
            print()
        print(sum(train_loss) / len(train_loss))

    def validate(self, data_loader, device="cpu"):
        loss_fnc = nn.NLLLoss()
        loss_lst = []
        self.eval()
        pass

    def predict(self, string):
        """

        """
        self.eval()

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


