import argparse
import numpy as np
from rnn_classifier import *
from rnn_dataset import *
from data_utils import Vocabulary
import torch
import os
from torch.nn.utils.rnn import pad_sequence

class Tagger:
    def __init__(self, mod_dir, filename, modname, toksfile, tagsfile):

        self.model, self.toks_vocab, self.tags_vocab = MweRNN.load(os.path.join(mod_dir, filename), modname,
                                                                   os.path.join(mod_dir, toksfile),
                                                                   os.path.join(mod_dir, tagsfile), embsize=64, hidden_size=64, drop_out=0.1, device="cpu")

        self.model.to("cpu")
        self.model.eval()

    def __call__(self, sentences):
        """
        predict the tags given a batch of sentence
        :param sentences: a list of string represent a sentence
        :return:
        """
        input = [torch.tensor([self.toks_vocab[tok] for tok in str(sent).split()]) for sent in sentences]

        X     = pad_sequence(input, batch_first = True, padding_value = self.toks_vocab["<pad>"])

        best_scores, best_paths= self.model.predict(X)
        out                    = []

        for path in best_paths:
            out.append(list(self.tags_vocab.rev_lookup(int(t)) for t in path))

        return out

    def _lexical_info(self):
        """
        add lexical information to the tagger
        :return:
        """
        pass


