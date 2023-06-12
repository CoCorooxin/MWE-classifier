from data_utils import Vocabulary
from models import AttentionRNN
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
import rnn_dataset
from crf import CRF
import os


class MweRNN(nn.Module):

    def __init__(self, name, toks_vocab, tags_vocab, emb_size=64, hidden_size=64, drop_out=0.):

        super(MweRNN, self).__init__()

        self.word_embedding = nn.Embedding(len(toks_vocab), emb_size)
        #self.depl_embedding = nn.Embedding(len(deprel_vocab), emb_size)

        self.toks_vocab   = toks_vocab
        self.tags_vocab   = tags_vocab
        self.padidx       = tags_vocab["<pad>"]

        self.relu         = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        self.crf = CRF(emb_size, self.tags_vocab)

        if name == "RNN":
            self.rnn = nn.RNN(emb_size, hidden_size // 2, batch_first=True, num_layers=1, bidirectional=True)
        if name == "LSTM":
            self.rnn = nn.LSTM(emb_size, hidden_size // 2, batch_first=True, num_layers=1, bidirectional=True)
        if name == "ATRNN":
            self.rnn = AttentionRNN(emb_size, hidden_size, drop_out = drop_out)


    def forward(self, Xtoks_IDs):

        emb = self.word_embedding(Xtoks_IDs)#+ self.depl_embedding(deprel)  # Batch, inputsize*emb_size
        #print(emb.shape) #B, window_size, emb_size
        # input.view(b, -1) B, window_size * emb_size
        logits, _ = self.rnn(emb)

        masks = (Xtoks_IDs == self.toks_vocab["<pad>"])
        #print(output.shape)
        return logits, (~masks) # bs, seq, embsize

    def _init_weights(self):
        pass

    def train_model(self, train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", split_train=0.8):
        """
        the train data is in form of nested lists: [sentences[tokens]]
        """
        self.to(device)
        # adaptive gradient descent, for every update lr is a function of the amount of change in the parameters
        optimizer   = torch.optim.Adam(self.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        loss_fnc = nn.NLLLoss()
        test_loader = test_data.get_loader(batch_size=batch_size)
        num_train_examples = int(split_train * len(train_data))
        trainset, validset = random_split(train_data, [num_train_examples, len(train_data) - num_train_examples])
        train_loader = rnn_dataset.Mysubset(trainset, self.toks_vocab, self.tags_vocab).get_loader(batch_size=batch_size, shuffle=True)
        dev_loader = rnn_dataset.Mysubset(validset, self.toks_vocab, self.tags_vocab).get_loader(batch_size=batch_size,shuffle=False)

        train_loss = []

        for e in range(epochs):
            self.train()
            ep_loss = []

            for X_toks, Y_gold in tqdm(train_loader):
                bs, seq = X_toks.shape
                optimizer.zero_grad()
                logits, masks = self.forward(X_toks)
                #print(X_toks.shape)
                loss =  self.crf.loss(logits, Y_gold, masks) #conditional random field
                #loss_value = loss_fnc(logprobs.view(bs*seq, -1), Y_gold.view(-1))
                ep_loss.append(loss.mean().item())
                loss.backward(loss)
                optimizer.step()

            loss = sum(ep_loss) / len(ep_loss)
            train_loss.append(loss)
            valid_loss = self.validate(dev_loader)

            # print("Epoch %d | Mean train loss  %.4f | Mean dev loss %.4f"%(e,loss, devloss) )
            print("Epoch %d | Mean train loss  %.4f |  Mean dev loss  %.4f " % (e, loss, valid_loss))
            print()

        TP, FP, FN, average_precision, average_recall, average_f1_score, weighted_f1_score, weighted_recall, weighted_precision = self.evaluate(test_loader)
        print("AVR: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (average_precision, average_recall, average_f1_score))
        print("Weighted: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (weighted_f1_score, weighted_recall, weighted_precision))

    def validate(self, data_loader, device="cpu"):
        self.to(device)
        loss_lst = []
        self.eval()
        with torch.no_grad():
            for X_toks, Y_gold in tqdm(data_loader):
                bs, seq = X_toks.shape
                logits, masks = self.forward(X_toks)
                #loss = loss_fnc(logprobs.view(bs*seq, -1), Y_gold.view(-1))
                loss = self.crf.loss(logits , Y_gold , masks)

                loss_lst.append(loss)
        return sum(loss_lst) / len(loss_lst)

    def predict(self, batch_tensors, device = "cpu"):
        """
        :param batch_tensors: a batch tensor of input seqs
        :return: the predicted best path and the scores
        """
        self.eval()
        self.to(device)
        with torch.no_grad():
            logits, masks          = self.forward(batch_tensors)
            best_score, best_paths = self.crf(logits, masks)

        return best_score, best_paths

    def evaluate(self, test_loader, device = "cpu"):
        """
        evaluation the classifier with confusion matrix : precision recall and f-score
        """
        self.to(device)
        self.eval()
        num_tags = len(self.tags_vocab)
        # print(num_tags)
        TP = torch.zeros(num_tags)
        FP = torch.zeros(num_tags)
        FN = torch.zeros(num_tags)
        class_counts = torch.zeros(num_tags)
        with torch.no_grad():
            for X_toks, Y_golds in tqdm(test_loader):
                # Forward pass
                logits, masks = self.forward(X_toks)
                best_score, best_paths = self.crf(logits, masks) #viterbi
                #print(best_paths.shape)
                for i in range(len(best_paths)):
                    path = torch.tensor(best_paths[i])
                    gold = torch.tensor([j for j in Y_golds[i] if j != self.padidx])
                    for tag in path:
                        TP[tag] += ((path == tag) & (gold == tag)).sum()
                        FP[tag] += ((path == tag) & (gold != tag)).sum()
                        FN[tag] += ((path != tag) & (gold == tag)).sum()
                        class_counts[tag] += (gold == tag).sum()

        # Calculate precision, recall, and F1 score for each tag
        precision = TP / (TP + FP)
        # avoid nan
        nan_mask = torch.isnan(precision)
        precision[nan_mask] = 0.

        recall = TP / (TP + FN)
        # avoid nan
        nan_mask = torch.isnan(recall)
        recall[nan_mask] = 0.

        f1_score = 2 * (precision * recall) / (precision + recall)
        # avoid nan
        nan_mask = torch.isnan(f1_score)
        f1_score[nan_mask] = 0.
        # Calculate class weights
        class_weights = class_counts / class_counts.sum()

        # Calculate average precision, recall, and F1 score
        average_precision = torch.mean(precision)
        average_recall = torch.mean(recall)
        average_f1_score = torch.mean(f1_score)

        weighted_f1_score = torch.sum(f1_score * class_weights)
        weighted_recall = torch.sum(recall * class_weights)
        weighted_precision = torch.sum(precision * class_weights)

        return TP, FP, FN, average_precision, average_recall, average_f1_score, weighted_f1_score, weighted_recall, weighted_precision

    @staticmethod
    def load(modelfile,name, toks_vocab, tags_vocab, embsize, hidden_size, drop_out, device):
        toks_vocab = Vocabulary.read(toks_vocab)
        tags_vocab = Vocabulary.read(tags_vocab)
        model      = MweRNN(name, toks_vocab,tags_vocab, embsize, hidden_size, drop_out)
        model.load_params(modelfile,device)
        return model,toks_vocab, tags_vocab

    def load_params(self,param_filename,device):
        self.load_state_dict(torch.load(param_filename, map_location=device))

    def save(self, path, name):
        self.toks_vocab.write(os.path.join(path, "toks.txt"))
        self.tags_vocab.write(os.path.join(path, "tags.txt"))
        torch.save(self.state_dict(), os.path.join(path, name))


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

    model, toks_vocab, tags_vocab = MweRNN.load(config['MODEL_DIR'],config['NAME'], toks_vocab,tags_vocab, config['EMBSIZE'],config['HIDDENSIZE'], config['DROPOUT'], config["DEVICE"])
    #train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", reg=None, split_train=0.8):

    model.train_model(config["TRAIN"], config["TEST", config["EPOCHS"], config["LR"], config["SPLIT"], config["DEVICE"]])

    model.save("trained_models", "rnn_mod.pth")
