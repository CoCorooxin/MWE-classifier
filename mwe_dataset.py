from torch.utils.data import Dataset, DataLoader
import torch

class Vocabulary:

    def __init__(self, symbols = None):

        # dictionary to map the vocabulary with a id to build matrix
        # add "UNK" for unknown word to the initial mapping
        self.word2idx = dict()
        self.idx2word = []

        if symbols:
            for sym in symbols:
                self.update(sym)

    def update(self, tok):

        # takes as input a symbol and build the mapping if it doesnt exist
        if tok not in self.word2idx:
            self.word2idx[tok] = len(self.idx2word)
            self.idx2word.append(tok)

    def lookup(self, tok, update = False):

        # find tok id given the string, if the tok does not exist return the idx of "UNK"
        if tok not in self.word2idx:
            if update:
                self.update(tok)
                return self[tok]
            return self.word2idx["<unk>"]

        return self.word2idx[tok]
    def rev_lookup(self, idx):

        # find the tok string given the id
        return self.idx2word[idx]

    def __getitem__(self, symbol):

        # if the symbol does not exist we see it as unk
        return self.lookup(symbol)

    def __len__(self):

        return len(self.idx2word)


"""
Functions for reading and writing UD CONLL data
"""
CONLL_FIELDS = ["token", "pos", "features", "deprel"]
MWE_TAGS = ["outside", "mwehead", "component"]


def readfile(filename, update=False, tok_vocab=None, pos_vocab=None, mwe_vocab=None, deprel_vocab=None):
    """
    function to read and encode the corpus at one pass
    """

    istream = open(filename, encoding="utf-8")
    x_toks, pos_tags, deprels, y_mwe = [], [], [], []

    for line in istream:
        line = line.strip()
        if line and line[0] != "#":
            try:
                tokidx, token, lemma, upos, pos, features, headidx, deprel, extended, _ = line.split()

            except ValueError:
                pass
            if tokidx == "1":
                # beginning of sentence
                x_toks.append(tok_vocab["<bos>"])
                y_mwe.append(mwe_vocab["outside"])
                deprels.append(deprel_vocab["<unk>"])
                pos_tags.append(pos_vocab["<unk>"])

            # extract info
            x_toks.append(tok_vocab.lookup(tok=token, update=update))
            pos_tags.append(pos_vocab.lookup(pos, update=update))
            deprels.append(deprel_vocab.lookup(deprel, update=update))
            # make simple mwe tags
            if features.startswith("mwehead"):
                y_mwe.append(mwe_vocab["mwehead"])
            elif features.startswith("component"):
                y_mwe.append(mwe_vocab["component"])
            else:
                y_mwe.append(mwe_vocab["outside"])

        elif len(line) == 0 and x_toks:
            # end of sentence
            x_toks.append(tok_vocab["<eos>"])
            y_mwe.append(mwe_vocab["outside"])
            deprels.append(deprel_vocab["<unk>"])
            pos_tags.append(pos_vocab["<unk>"])

    # end of corpus
    x_toks.append(tok_vocab["<eos>"])
    y_mwe.append(mwe_vocab["outside"])
    deprels.append(deprel_vocab["<unk>"])
    pos_tags.append(pos_vocab["<unk>"])

    istream.close()

    return x_toks, pos_tags, y_mwe, deprels
# [{"token1": "token", "multiword": "mwe", "mwe lemma": "mwe lemma"}, {"token2": "token", "multiword": "mwe"}, {"token3": "token", "multiword": "mwe"}]


class MWEDataset(Dataset):

    def __init__(self, datafilename=None, lst_toks=None, tok_vocab=None, pos_vocab=None, mwe_vocab=None,
                 deprel_vocab=None, isTrain=False, context_size=1):
        """
        take as input either the path to a conllu file or a list of tokens
        we consider context size as the n preceding and n subsequent words in the text as the context for predicting the next word.
        """
        super(MWEDataset, self).__init__()

        self.tok_vocab, self.pos_vocab, self.mwe_vocab, self.deprel_vocab = tok_vocab, pos_vocab, mwe_vocab, deprel_vocab

        if datafilename:
            self.x_toks, self.pos_tags, self.y_mwe, self.deprels = readfile("corpus/train.conllu",
                                                                             update=isTrain,
                                                                             tok_vocab=self.tok_vocab,
                                                                             pos_vocab=self.pos_vocab,
                                                                             mwe_vocab=self.mwe_vocab,
                                                                             deprel_vocab=self.deprel_vocab)
        elif lst_toks:
            self.x_toks = lst_toks
            self.tok_vocab = Vocabulary(lst_toks)
            self.pos_tags = [0] * len(self.x_toks)
            self.deprels = [0] * len(self.x_toks)
            self.pos_vocab = Vocabulary(["<unk>"])
            self.deprel_vocab = Vocabulary(["<unk>"])

        print('token Vocab size', len(self.tok_vocab))
        self.context_size = context_size
        self.context = [0] * self.context_size + self.x_toks + [0] * self.context_size
        self.ctxt_deprels = [0] * self.context_size + self.deprels + [0] * self.context_size
        self.ctxt_pos = [0] * self.context_size + self.pos_tags + [0] * self.context_size

    def __len__(self):
        return len(self.x_toks)

    def __getitem__(self, idx):
        """
        return the X as the concatenation of token, and the tokens in the immediate context window
        Y is a real number representing the idx for mwe tags
        """

        X = torch.tensor(self.context[idx: idx + self.context_size] +
                         [self.x_toks[idx]] +
                         self.context[idx + self.context_size + 1: idx + 2 * self.context_size + 1])

        X_deprel = torch.tensor(self.ctxt_deprels[idx: idx + self.context_size] +
                                [self.deprels[idx]] +
                                self.ctxt_deprels[idx + self.context_size + 1: idx + 2 * self.context_size + 1])

        X_pos = torch.tensor(self.ctxt_pos[idx: idx + self.context_size] +
                             [self.pos_tags[idx]] +
                             self.ctxt_pos[idx + self.context_size + 1: idx + 2 * self.context_size + 1])
        Y_true = self.y_mwe[idx]

        return X, X_deprel, X_pos, Y_true

    def as_strings(self, batch_tensor):
        """
        Returns a string representation of a tensor of word indexes
        """
        out = []
        for line in batch_tensor.tolist():
            out.append([self.tok_vocab.rev_lookup(idx) for idx in line])
        return out

    def get_loader(self, batch_size=1, num_workers=0, word_dropout=0., shuffle=False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
