from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
import itertools
from data_utils import Vocabulary, upos2pos

"""
Functions for reading and writing UD CONLL data
"""
CONLL_FIELDS = ["token", "pos", "features", "deprel"]
MWE_TAGS = ["B", "I"]  # B for begin , I for inside


def readfile(filename, update=False, toks_vocab=Vocabulary(["<unk>", "<pad>"]),
             tags_vocab=Vocabulary(["<unk>", "<pad>"])):
    """
    function to read the corpus at one pass
    signature for train corpus : X_toks, Y_tags = readfile("corpus/train.conllu", update=True)
    signature for test corpus/ dev corpus:  X_test, Y_test = readfile("corpus/train.conllu", update=True, vocabtoks_train, vocabtags_train)
    """

    istream = open(filename, encoding="utf-8")
    X_toks, Y_tags = [], []
    sent_toks, sent_tags = [], []

    for line in istream:
        line = line.strip()
        if line and line[0] != "#":
            try:
                tokidx, token, lemma, pos, upos, features, headidx, deprel, extended, _ = line.split()

            except ValueError:
                pass

            mwe_tag = lambda x: "I" if features.startswith("component") else "B"
            # extract tagging information
            tag = mwe_tag(features) + "_" + upos2pos(pos)
            sent_toks.append(token)
            sent_tags.append(tag)

            if update:
                toks_vocab.update(token)
                tags_vocab.update(tag)

        elif sent_toks:

            X_toks.append(sent_toks)
            Y_tags.append(sent_tags)
            sent_toks, sent_tags = [], []

    istream.close()
    # return the encoded data in list of list, the nested list represents the sentences
    return X_toks, Y_tags, toks_vocab, tags_vocab
# [{"token1": "token", "multiword": "mwe", "mwe lemma": "mwe lemma"}, {"token2": "token", "multiword": "mwe"}, {"token3": "token", "multiword": "mwe"}]

class MWEDataset(Dataset):

    def __init__(self, datafilename=None, toks_vocab=Vocabulary(["<unk>", "<pad>"]),
                 tags_vocab=Vocabulary(["<unk>", "<pad>"]), isTrain=False, window_size=0):
        """
        take as input either the path to a conllu file or a list of tokens
        we consider context size as the n preceding and n subsequent words in the text as the context for predicting the next word.
        """
        super(MWEDataset, self).__init__()

        self.toks_vocab, self.tags_vocab = toks_vocab, tags_vocab
        if datafilename:
            self.Xtoks, self.Ytags, self.toks_vocab, self.tags_vocab = readfile(datafilename,
                                                                                update=isTrain,
                                                                                toks_vocab=toks_vocab,
                                                                                tags_vocab=tags_vocab)
            self.window_size = window_size
            self.data = self.build_dataset()
            self.tags_dist = Counter(itertools.chain(*self.Ytags))

        print('token Vocab size', len(self.toks_vocab))

    def __len__(self):
        return len(self.data)

    def build_dataset(self):
        """
        build fixed length examples with contextual tokens as features
        takes as input a nested list of encoded corpus, [sentences[tokens]]
        return a list of examples with context window features
        """
        # print(X_toks)
        examples = []
        for i in range(len(self.Xtoks)):
            toks = ["<pad>"] * self.window_size + self.Xtoks[i] + ["<pad>"] * self.window_size  # 3+3+3
            for j in range(self.window_size, len(toks) - self.window_size, 1):
                examples.append((toks[j - self.window_size: j + self.window_size + 1], self.Ytags[i][j - self.window_size]))

        # print(examples[-1])
        return examples

    def __getitem__(self, idx):
        return self.data[idx]

    def as_strings(self, batch_tensor):
        """
        Returns a string representation of a tensor of word indexes
        """
        out = []
        for line in batch_tensor.tolist():
            out.append([self.toks_vocab.rev_lookup(idx) for idx in line])
        return out

    def get_loader(self, batch_size=1, num_workers=0, shuffle=False):
        def mk_batch(selected_items):
            XtoksIDs = []
            YtagsIDs = []
            for tokens, tag in selected_items:
                x = torch.tensor([self.toks_vocab[tok] for tok in tokens])
                y = torch.tensor(self.tags_vocab[tag])

                XtoksIDs.append(x)
                YtagsIDs.append(y)
            return torch.stack(XtoksIDs) , torch.stack(YtagsIDs)

        # specify that the mk_batch function should be used to collate the individual samples into batches.
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=mk_batch)


class Mysubset(MWEDataset):
    """
    an auxiliary class to take care of the K fold validation, split the train corpus into train and dev
    """

    def __init__(self, subset, toks_vocab, tags_vocab):
        self.subset = subset
        self.toks_vocab = toks_vocab
        self.tags_vocab = tags_vocab

    def __getitem__(self, index):
        return self.subset[index]

    def __len__(self):
        return len(self.subset)