from mwe_dataset import MWEDataset, Vocabulary, Mysubset
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import rnn_dataset
from torchcrf import CRF
import numpy as np
from mlp_baseline import MLP_baseline,AttentionMLP
import os
class MweRNN(nn.Module):

    def __init__(self, name, toks_vocab, tags_vocab, deprel_vocab, window_size=0, emb_size=64, hidden_size=64, pretrainedw2v=None, drop_out=0.):

        super(MweRNN, self).__init__()

        self.word_embedding = nn.Embedding(len(toks_vocab), emb_size)
        #self.depl_embedding = nn.Embedding(len(deprel_vocab), emb_size)
        self.window_size  = window_size
        self.input_length = 1 + window_size * 2
        self.toks_vocab   = toks_vocab
        self.tags_vocab   = tags_vocab
        self.deprel_vocab = deprel_vocab

        self.FFW          = nn.Linear(hidden_size , len(tags_vocab))  # output # of classes
        self.logsoftmax   = nn.LogSoftmax(dim=1)
        self.relu         = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        #self.attention = AttentionMLP(hidden_size, drop_out= drop_out)  # Add attention layer
        self.crf = CRF(len(tags_vocab))

        self.rnn = nn.RNN(emb_size,hidden_size, batch_first = True, num_layers=1)

        if pretrainedw2v:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.pretrainedw2v_loader(pretrainedw2v).wv.vectors))

    def forward(self, Xtoks_IDs,deprel=None):

        emb = self.word_embedding(Xtoks_IDs)#+ self.depl_embedding(deprel)  # Batch, inputsize*emb_size
        #print(emb.shape) #B, window_size, emb_size
        # input.view(b, -1) B, window_size * emb_size

        logits, hid = self.rnn(emb)
        #attn_output = self.attention(logits)
        output = self.logsoftmax(self.FFW(self.dropout(logits)))  #bs, seq*embsize
        #print(output.shape)
        return  output # bs, seq, embsize

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

        train_loss = []

        for e in range(epochs):
            self.train()
            # at every epochs we split the traindata into train set and dev set
            num_train_examples = int(split_train * len(train_data))
            trainset, validset = random_split(train_data, [num_train_examples, len(train_data) - num_train_examples])
            train_loader = rnn_dataset.Mysubset(trainset, self.toks_vocab, self.tags_vocab, self.deprel_vocab).get_loader(batch_size=batch_size, shuffle=True)
            dev_loader =  rnn_dataset.Mysubset(validset, self.toks_vocab, self.tags_vocab, self.deprel_vocab).get_loader(batch_size=batch_size, shuffle=False)
            ep_loss = []

            for X_toks, depl, Y_gold in tqdm(train_loader):
                bs, seq = X_toks.shape
                optimizer.zero_grad()
                #print(X_toks.shape)
                logprobs = self.forward(X_toks)
                #print(logprobs.shape)
                mask = (Y_gold == train_data.tag_padidx)
                loss = -self.crf(logprobs, Y_gold, (~mask)) #conditional random field
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

        average_precision, average_recall, average_f1_score, weighted_f1_score, weighted_recall, weighted_precision = self.evaluate(test_loader)
        print("AVR: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (average_precision, average_recall, average_f1_score))
        print("Weighted: Precision %.4f | Recall  %.4f |  F-score  %.4f " % (weighted_f1_score, weighted_recall, weighted_precision))

    def validate(self, data_loader, device="cpu"):
        loss_fnc = nn.NLLLoss()
        loss_lst = []
        self.eval()
        with torch.no_grad():
            for X_toks, deprel, Y_gold in tqdm(data_loader):
                bs, seq = X_toks.shape
                logprobs = self.forward(X_toks)

                #loss = loss_fnc(logprobs.view(bs*seq, -1), Y_gold.view(-1))
                loss = -self.crf(logprobs.view(bs * seq, -1), Y_gold.view(-1))
                loss_lst.append(loss)
        return sum(loss_lst) / len(loss_lst)

    def predict(self, testset = None, input_str = None):
        """

        """
        self.eval()
        with torch.no_grad():
            for X_toks, deprel, Y_gold in tqdm(testset.get_loader(batch_size = 100)):
                logprobs = self.forward(X_toks,deprel)
                # Decode the best tag sequence using Viterbi decoding
                best_path = self.crf.decode(logprobs)



    def evaluate(self, test_loader):
        """
        evaluation the classifier with confusion matrix : precision recall and f-score
        """
        self.eval()
        num_tags = len(self.tags_vocab)
        # print(num_tags)
        TP = torch.zeros(num_tags)
        FP = torch.zeros(num_tags)
        FN = torch.zeros(num_tags)
        class_counts = torch.zeros(num_tags)
        with torch.no_grad():
            for X_toks, deprel, Y_golds in tqdm(test_loader):
                # Forward pass
                logprobs = self.forward(X_toks)
                best_path = self.crf.decode(logprobs) #viterbi

                # Mask out the padding positions
                mask = (X_toks != self.toks_vocab["pad"])
                best_path = best_path[mask]
                Y_golds = Y_golds[mask]

                # Update confusion matrix
                for tag in range(num_tags):
                    TP[tag] += ((best_path == tag) & (Y_golds == tag)).sum()
                    FP[tag] += ((best_path == tag) & (Y_golds != tag)).sum()
                    FN[tag] += ((best_path != tag) & (Y_golds == tag)).sum()
                    class_counts[tag] += (Y_golds == tag).sum()
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

        return average_precision, average_recall, average_f1_score, weighted_f1_score, weighted_recall, weighted_precision

    @staticmethod
    def load(modelfile,name, toks_vocab, tags_vocab, window_size, embsize, hidden_size, drop_out, device):
        toks_vocab = Vocabulary.read(toks_vocab)
        tags_vocab = Vocabulary.read(tags_vocab)
        model      = MweClassifier(name, toks_vocab,tags_vocab, window_size, embsize, hidden_size, drop_out)
        model.load_params(modelfile,device)
        return model,toks_vocab, tags_vocab

    def load_params(self,param_filename,device):
        self.load_state_dict(torch.load(param_filename, map_location=device))

    def save(self, path, name):
        self.toks_vocab.write(os.path.join(path, "toks.vocab"))
        self.tags_vocab.write(os.path.join(path, "tags.vocab"))
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

    model, toks_vocab, tags_vocab = MweClassifier.load(config['MODEL_DIR'],config['NAME'], toks_vocab,tags_vocab, config['WINDOW_SIZE'], config['EMBSIZE'],config['HIDDENSIZE'], config['DROPOUT'], device=args.device)
    #train_data, test_data, epochs=10, lr=1e-3, batch_size=10, device="cpu", reg=None, split_train=0.8):

    model.train_model(config["TRAIN"], config["TEST", config["EPOCHS"], config["'"]])

    model.save("trained_models")
