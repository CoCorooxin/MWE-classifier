from mwe_dataset import MWEDataset
import torch
from collections import Counter
import torch.nn as nn
from tqdm import tqdm
import numpy as np


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

    def train_model(self, train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", reg=None,
                    split_train=0.8):
        """
        the train data is in form of nested lists: [sentences[tokens]]
        """
        self.to(device)
        # adaptive gradient descent, for every update lr is a function of the amount of change in the parameters
        # optimizer   = torch.optim.Adam()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        loss_fnc = nn.NLLLoss()
        test_loader = test_data.get_loader(batch_size=batch_size)

        train_loss = []

        for e in range(epochs):
            self.train()
            # at every epochs we split the traindata into train set and dev set
            num_train_examples = int(split_train * len(train_data))
            trainset, validset = random_split(train_data, [num_train_examples, len(train_data) - num_train_examples])
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            dev_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)

            ep_loss = []

            for X_toks, Y_gold in tqdm(train_loader):
                # print(x.shape)
                optimizer.zero_grad()
                logprobs = self.forward(X_toks)

                # print(y_hat.shape)
                loss_value = loss_fnc(logprobs, Y_gold)
                ep_loss.append(loss_value.item())
                loss_value.backward()
                optimizer.step()
            loss = sum(ep_loss) / len(ep_loss)
            train_loss.append(loss)
            valid_loss = self.validate(dev_loader)

            # print("Epoch %d | Mean train loss  %.4f | Mean dev loss %.4f"%(e,loss, devloss) )
            print("Epoch %d | Mean train loss  %.4f |  Mean dev loss  %.4f " % (e, loss, valid_loss))
            print()

        average_precision, average_recall, average_f1_score = self.evaluation(test_loader)
        print("Precision %.4f | Recall  %.4f |  F-score  %.4f " % (average_precision, average_recall, average_f1_score))

    def validate(self, data_loader, device="cpu"):
        loss_fnc = nn.NLLLoss()
        loss_lst = []
        self.eval()
        with torch.no_grad():
            for X_toks, Y_gold in tqdm(data_loader):
                logprobs = self.forward(X_toks)

                loss = loss_fnc(logprobs, Y_gold)
                loss_lst.append(loss)
        return sum(loss_lst) / len(loss_lst)

    def predict(self, string):
        """

        """
        self.eval()

        pass

    def evaluation(self, test_loader):
        """
        evaluation the classifier with confusion matrix : precision recall and f-score
        """
        self.eval()
        num_tags = len(self.tags_vocab)
        TP = torch.zeros(num_tags)
        FP = torch.zeros(num_tags)
        FN = torch.zeros(num_tags)
        with torch.no_grad():
            for X_toks, Y_golds in tqdm(test_loader):
                logprobs = self.forward(X_toks)
                scores, predicted_IDs = torch.max(logprobs.data, dim=1)
                # convert tensor to np arrays
                predicted_IDs = predicted_IDs.cpu().numpy()
                Y_golds = Y_golds.cpu().numpy()
                for tag in range(num_tags):
                    TP[tag] += ((predicted_IDs == tag) & (Y_golds == tag)).sum()
                    FP[tag] += ((predicted_IDs == tag) & (Y_golds != tag)).sum()
                    FN[tag] += ((predicted_IDs != tag) & (Y_golds == tag)).sum()
        # Calculate precision, recall, and F1 score for each tag
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Calculate average precision, recall, and F1 score
        average_precision = torch.mean(precision)
        average_recall = torch.mean(recall)
        average_f1_score = torch.mean(f1_score)

        return average_precision, average_recall, average_f1_score

    def load_model(self, modelpath):
        pass



