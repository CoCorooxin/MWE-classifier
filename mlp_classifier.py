from mlp_dataset import MWEDataset
from torch.utils.data import Dataset, DataLoader, random_split
from data_utils import Vocabulary
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from models import MLP_baseline
from gensim.models import KeyedVectors, Word2Vec

class MLPClassifier(nn.Module):

    def __init__(self, toks_vocab, tags_vocab, window_size=0, emb_size=200, hidden_size=64, drop_out=0., pretrained = False, device= 'cpu'):

        super(MLPClassifier, self).__init__()
        self.window_size = window_size
        self.input_length = 1 + window_size * 2
        self.toks_vocab = toks_vocab
        self.tags_vocab = tags_vocab
        self.emb_size  = emb_size
        self.device   = device

        self.word_embedding = nn.Embedding(len(toks_vocab), emb_size).to(device)
        if pretrained:
            self._load_pretrained(device)
        self.FFW          = nn.Linear(hidden_size , len(tags_vocab))  # output # of classes
        self.logsoftmax   = nn.LogSoftmax(dim=1)

        self.net = MLP_baseline(emb_size= emb_size*(window_size*2+1), hidden_size=hidden_size, drop_out = drop_out)

    def forward(self, Xtoks_IDs):
        bs, seq = Xtoks_IDs.shape
        emb = self.word_embedding(Xtoks_IDs) # Batch, inputsize*emb_size
        #print(emb.shape) #B, window_size, emb_size
        # input.view(b, -1) B, window_size * emb_size
        logits = self.net(emb.view(bs, -1))
        output = self.logsoftmax(self.FFW(logits))  #bs, seq*embsize
        #print(output.shape)
        return  output # bs, seq, embsize

    def _load_pretrained(self):
        word_vectors = KeyedVectors.load_word2vec_format("corpus/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", binary=True, limit=500000)
        pretrained_weights = []
        for idx,word in enumerate(self.toks_vocab.idx2word):
            if word in word_vectors:
                pretrained_weights.append(torch.tensor(word_vectors[word]))
            else:
                pretrained_weights.append(torch.FloatTensor(self.emb_size).uniform_(-0.25, 0.25))  # Randomly initialize for unknown words
        pretrained_weights = torch.stack(pretrained_weights)
        self.word_embedding.weight.data.copy_(pretrained_weights)
        self.word_embedding.weight.requires_grad = False

    def train_model(self, train_data, test_data, dev_data, epochs=10, lr=1e-3, batch_size=10, device="cpu"):
        """
        the train data is in form of nested lists: [sentences[tokens]]
        """
        self.to(device)
        # adaptive gradient descent, for every update lr is a function of the amount of change in the parameters
        optimizer   = torch.optim.Adam(self.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        loss_fnc = nn.NLLLoss()

        test_loader, dev_loader = test_data.get_loader(batch_size=batch_size*10), dev_data.get_loader(batch_size=batch_size*10)
        train_loader            = train_data.get_loader(batch_size=batch_size)

        train_loss = []

        for e in range(epochs):
            self.train()
            ep_loss = []

            for X_toks, Y_gold in tqdm(train_loader):
                # print(x.shape)
                optimizer.zero_grad()
                logprobs = self.forward(X_toks.to(device))
                # print(y_hat.shape)
                loss_value = loss_fnc(logprobs, Y_gold.to(device))
                ep_loss.append(loss_value.item())
                loss_value.backward()
                optimizer.step()
            loss = sum(ep_loss) / len(ep_loss)
            train_loss.append(loss)
            valid_loss = self.validate(dev_loader, device = device)

            # print("Epoch %d | Mean train loss  %.4f | Mean dev loss %.4f"%(e,loss, devloss) )
            print("Epoch %d | Mean train loss  %.4f |  Mean dev loss  %.4f " % (e, loss, valid_loss))
            print()

        class_counts, TP, FP, FN, average_precision, average_recall, average_f1_score, weighted_f1_score, weighted_recall, weighted_precision = self.evaluate(test_loader, device = device)
        print("AVR: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (average_precision, average_recall, average_f1_score))
        print("Weighted: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (weighted_f1_score, weighted_recall, weighted_precision))

    def validate(self, data_loader, device="cpu"):
        loss_fnc = nn.NLLLoss()
        loss_lst = []
        self.eval()
        self.to(device)
        with torch.no_grad():
            for X_toks, Y_gold in tqdm(data_loader):
                logprobs = self.forward(X_toks.to(device))
                loss = loss_fnc(logprobs, Y_gold.to(device))
                loss_lst.append(loss)
        return sum(loss_lst) / len(loss_lst)

    def evaluate(self, test_loader, device = 'cpu'):
        """
        evaluation the classifier with confusion matrix : precision recall and f-score
        """
        self.eval()
        self.to(device)
        num_tags = len(self.tags_vocab)
        # print(num_tags)
        TP = torch.zeros(num_tags)
        FP = torch.zeros(num_tags)
        FN = torch.zeros(num_tags)
        class_counts = torch.zeros(num_tags)
        with torch.no_grad():
            for X_toks, Y_gold in tqdm(test_loader):
                logprobs = self.forward(X_toks.to(device))
                scores, predicted_IDs = torch.max(logprobs.data, dim=1)
                # convert tensor to np arrays
                predicted_IDs = predicted_IDs.cpu().numpy()
                Y_gold = Y_gold.cpu().numpy()
                for tag in range(num_tags):
                    TP[tag] += ((predicted_IDs == tag) & (Y_gold == tag)).sum()
                    FP[tag] += ((predicted_IDs == tag) & (Y_gold != tag)).sum()
                    FN[tag] += ((predicted_IDs != tag) & (Y_gold == tag)).sum()
                    class_counts[tag] += (Y_gold == tag).sum()
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

        return class_counts, TP, FP, FN, average_precision, average_recall, average_f1_score, weighted_f1_score, weighted_recall, weighted_precision

    @staticmethod
    def load(modelfile, toks_vocab, tags_vocab, window_size, embsize, hidden_size, drop_out,  pretrained = False, device="cpu"):
        toks_vocab = Vocabulary.read(toks_vocab)
        tags_vocab = Vocabulary.read(tags_vocab)
        model      = MLPClassifier(toks_vocab,tags_vocab, window_size, embsize, hidden_size, drop_out,  pretrained, device)
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
    lr = float(config["LR"])
    embsize = int(config["EMBSIZE"])
    hidsize = int(config["HIDDENSIZE"])
    bs      = int(config["BATCHSIZE"])
    winsize = int(config["WINDOW_SIZE"])
    epochs  = int(config["EPOCHS"])
    pretrain= config["PRETRAINED"]
    device  = config["DEVICE"]
    dropout = float(config["DROPOUT"])
    train   = MWEDataset(config["TRAIN"], window_size = winsize, isTrain = True)
    test    = MWEDataset(config["TEST"], window_size = winsize, isTrain = False)
    dev     = MWEDataset(config["DEV"], window_size = winsize)

    cstream.close()
    toks_vocab  = train.toks_vocab
    tags_vocab  = train.tags_vocab
    model       = MLPClassifier(toks_vocab, tags_vocab,winsize, embsize,hidsize, dropout,pretrain, device)
    #train_data, test_data, epochs=10, lr
    #(self, train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", split_train=0.8):
    model.train_model(train, test, dev, epochs, lr, bs,config["DEVICE"])

    model.save(config["MODEL_DIR"], config["MODEL_FILE"])