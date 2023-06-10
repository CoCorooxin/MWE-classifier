from mwe_dataset import MWEDataset, Vocabulary
import torch
import torch.nn.functional as F
from collections import Counter
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden, logits):
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        activate = torch.tanh(self.W1(logits) + self.W2(hidden))  # (batch_size, seq_length, hidden_size)
        attention_weights = F.softmax(self.V(activate), dim=1)  # (batch_size, seq_length, 1)
        new_logits = torch.sum(attention_weights * logits, dim=1)  # (batch_size, hidden_size)
        return new_logits

class MweClassifier(nn.Module):

    def __init__(self, model_name, toks_vocab, tags_vocab, window_size=0, emb_size=64, hidden_size=64, pretrainedw2v=None, drop_out=0.):

        super(MweClassifier, self).__init__()

        self.word_embedding = nn.Embedding(len(toks_vocab), emb_size)
        self.window_size  = window_size
        self.input_length = 1 + window_size * 2
        self.toks_vocab   = toks_vocab
        self.tags_vocab   = tags_vocab

        self.FFW          = nn.Linear(hidden_size * 2, len(tags_vocab)),  # output # of classes


        if pretrainedw2v:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.pretrainedw2v_loader(pretrainedw2v).wv.vectors))

    def forward(self, Xtoks_IDs):
        b, seq = Xtoks_IDs.shape

        input = self.word_embedding(Xtoks_IDs)  # Batch, inputsize*emb_size
        # print(input.shape) #B, window_size, emb_size
        # input.view(b, -1) B, window_size * emb_size
        nn.LogSoftmax(dim=1)
        return self.net(input.view(b, -1))  # tag_size = 3

    @staticmethod
    def pretrainedw2v_loader(self, path_to_pretrained=None):
        if not path_to_pretrained:
            # Download the pretrained French Word2Vec model https://fauconnier.github.io/#data
            model_name = 'frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin.gz'
            model = api.load(model_name)
        else:
            model = KeyedVectors.load_word2vec_format(path_to_pretrained, binary=True)
        return model

    def _init_weights(self):
        pass

    def train_model(self, train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", reg=None, split_train=0.8):
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

    @staticmethod
    def load(modelfile,toks_vocab, tags_vocab, window_size, embsize, hidden_size, drop_out, device):
        toks_vocab = Vocabulary.read(toks_vocab)
        tags_vocab = Vocabulary.read(tags_vocab)
        model      = MweClassifier(toks_vocab,tags_vocab, window_size, embsize, hidden_size, drop_out)
        model.load_params(modelfile,device)
        return model,toks_vocab, tags_vocab

    def load_params(self,param_filename,device):
        self.load_state_dict(torch.load(param_filename, map_location=device))

    def save(self, path):
        self.toks_vocab.write(os.path.join(path, "toks.vocab"))
        self.tags_vocab.write(os.path.join(path, "tags.vocab"))
        torch.save(self.state_dict(), path)


if __name__ == '__main__':
    import argparse
    import os
    import yaml
    parser = argparse.ArgumentParser("MWE Classifer")
    parser.add_argument('config_file')

    args = parser.parse_args()
    cstream = open(args.config_file)
    config = yaml.safe_load(cstream)
    cstream.close()
    toks_vocab  = Vocabulary.read(config["TOKS_VOCAB"])
    tags_vocab  = Vocabulary.read(config["TAGS_VOCAB"])

    model, toks_vocab, tags_vocab = MweClassifier.load(config['MODEL_DIR'], toks_vocab,tags_vocab, config['WINDOW_SIZE'], config['EMBSIZE'],config['HIDDENSIZE'], config['DROPOUT'], device=args.device)
    #train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", reg=None, split_train=0.8):

    model.train_model(config["TRAIN"], config["TEST", config["EPOCHS"], config["'"]])

    model.save("trained_models")
