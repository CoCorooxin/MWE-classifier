from data_utils import Vocabulary
from models import AttentionRNN, RNNmwe, LSTMmwe
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from crf import CRF
import os
from rnn_dataset import RnnDataset
from gensim.models import KeyedVectors, Word2Vec


class MweRNN(nn.Module):

    def __init__(self, name, toks_vocab, tags_vocab, emb_size=64, hidden_size=64, drop_out=0., pretrained = False, device = "cpu"):

        super(MweRNN, self).__init__()

        self.toks_vocab   = toks_vocab
        self.tags_vocab   = tags_vocab
        self.padidx       = tags_vocab["<pad>"]
        self.emb_size     = emb_size

        self.word_embedding = nn.Embedding(len(toks_vocab), emb_size).to(device)
        if pretrained:
            self._load_pretrained()

        self.relu         = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        if name == "RNN":
            self.rnn = RNNmwe(emb_size, hidden_size, device = device)
        if name == "LSTM":
            self.rnn = LSTMmwe(emb_size, hidden_size, device = device)
        if name == "ATRNN":
            self.rnn = AttentionRNN(emb_size, hidden_size, drop_out = drop_out, device = device)
        self.crf = CRF(hidden_size, self.tags_vocab)

    def forward(self, Xtoks_IDs, masks):

        emb = self.word_embedding(Xtoks_IDs)#+ self.depl_embedding(deprel)  # Batch, inputsize*emb_size
        #print(emb.shape) #B, window_size, emb_size
        # input.view(b, -1) B, window_size * emb_size
        logits, _ = self.rnn(emb, masks)   #B, seq, hiddensize

        #print(output.shape)
        return logits # bs, seq, embsize

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
        self.word_embedding.weight.requires_grad = True

    def train_model(self, train_data, test_data, dev_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", split =0.8):
        """
        the train data is in form of nested lists: [sentences[tokens]]
        """
        self.to(device)
        # adaptive gradient descent, for every update lr is a function of the amount of change in the parameters
        optimizer   = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        test_loader, dev_loader = test_data.get_loader(batch_size=batch_size * 10), dev_data.get_loader(batch_size=batch_size * 10)
        train_loader = train_data.get_loader(batch_size=batch_size)
        train_loss = []

        for e in range(epochs):
            self.train()
            ep_loss = []

            for X_toks, Y_gold in tqdm(train_loader):
                bs, seq = X_toks.shape
                optimizer.zero_grad()
                masks  = (X_toks != self.toks_vocab["<pad>"]).to(device)  #set padding idx to false
                logits = self.forward(X_toks.to(device), masks)
                #print(X_toks.shape)
                loss =  self.crf.loss(logits, Y_gold.to(device), masks) #conditional random field
                #loss_value = loss_fnc(logprobs.view(bs*seq, -1), Y_gold.view(-1))
                ep_loss.append(loss.mean().item())
                loss.backward(loss)
                optimizer.step()

            loss = sum(ep_loss) / len(ep_loss)
            train_loss.append(loss)
            valid_loss = self.validate(dev_loader, device = device)

            # print("Epoch %d | Mean train loss  %.4f | Mean dev loss %.4f"%(e,loss, devloss) )
            print("Epoch %d | Mean train loss  %.4f |  Mean dev loss  %.4f " % (e, loss, valid_loss))
            print()

        class_counts, TP, FP, FN, average_precision, average_recall, average_f1_score, weighted_f1_score, weighted_recall, weighted_precision = self.evaluate(test_loader)
        print("AVR: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (average_precision, average_recall, average_f1_score))
        print("Weighted: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (weighted_f1_score, weighted_recall, weighted_precision))

    def validate(self, data_loader, device="cpu"):
        self.to(device)
        loss_lst = []
        self.eval()
        with torch.no_grad():
            for X_toks, Y_gold in tqdm(data_loader):
                bs, seq = X_toks.shape
                masks = (X_toks != self.toks_vocab["<pad>"]).to(device) #create a mask where padidx is false
                logits = self.forward(X_toks.to(device), masks)
                #loss = loss_fnc(logprobs.view(bs*seq, -1), Y_gold.view(-1))
                loss = self.crf.loss(logits , Y_gold.to(device), masks)

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

            masks = (batch_tensors != self.toks_vocab["<pad>"]).to(device)
            logits          = self.forward(batch_tensors, masks)
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

                masks  = (X_toks != self.toks_vocab["<pad>"]).to(device)
                # Forward pass
                logits = self.forward(X_toks.to(device), masks)
                best_score, best_paths = self.crf(logits, masks) #viterbi
                best_paths = best_paths
                #print(best_paths.shape)
                for i in range(len(best_paths)):
                    path = torch.tensor(best_paths[i])
                    #print(Y_golds[i])
                    gold = torch.tensor([j for j in Y_golds[i] if j != self.padidx])
                    #print(path)
                    #print(gold)
                    for tag in range(num_tags):
                        TP[tag] += ((path == tag) & (gold == tag)).sum()
                        FP[tag] += ((path == tag) & (gold != tag)).sum()
                        FN[tag] += ((path != tag) & (gold == tag)).sum()
                        class_counts[tag] += (gold==tag).sum()

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
    lr = float(config["LR"])

    embsize = int(config["EMBSIZE"])
    hidsize = int(config["HIDDENSIZE"])
    bs      = int(config["BATCHSIZE"])
    epochs  = int(config["EPOCHS"])
    dropout = float(config["DROPOUT"])
    train   = RnnDataset(config["TRAIN"], isTrain = True)
    test    = RnnDataset(config["TEST"])
    dev     = RnnDataset(config["DEV"])
    device  = config["DEVICE"]
    pretrain= config["PRETRAINED"]

    cstream.close()
    toks_vocab  = train.toks_vocab
    tags_vocab  = train.tags_vocab
    model       = MweRNN(config['NAME'], toks_vocab, tags_vocab, embsize,hidsize, dropout, pretrained = pretrain, device = device)
    #train_data, test_data, epochs=10, lr
    #(self, train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", split=0.8):
    model.train_model(train, test, dev, epochs, lr, bs, device = device)


    model.save(config["MODEL_DIR"], config["MODEL_FILE"])

