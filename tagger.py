import argparse
import numpy as np
from rnn_classifier import *
from rnn_dataset import *

class Tagger:
    def __init__(self, model_dir, modname, toksfile, tagsfile):

        self.model = MweRNN.load(model_dir, modname, toksfile,tagsfile, embsize=64, hidden_size=64, drop_out=0.1, device="cpu")

        self.model.to("cpu")
        self.model.eval()
    def __call__(self, sentences, begin_tags):

