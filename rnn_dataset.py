from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.functional import pad
from collections import Counter
import itertools
from data_utils import Vocabulary,upos2pos

"""
Functions for reading and writing UD CONLL data
"""
CONLL_FIELDS = ["token", "pos", "features", "deprel"]
MWE_TAGS = ["B", "I"]  # B for begin , I for inside

def readfile(filename, update=False, toks_vocab=Vocabulary([ "<unk>", "<bos>", "<eos>", "<pad>"]), tags_vocab=Vocabulary([ "<unk>", "<bos>", "<eos>", "<pad>"]), deprel_vocab=Vocabulary([ "<unk>", "<bos>", "<eos>", "<pad>"])):
    """
    function to read the corpus at one pass
    signature for train corpus : X_toks, Y_tags = readfile("corpus/train.conllu", update=True)
    signature for test corpus/ dev corpus:  X_test, Y_test = readfile("corpus/train.conllu", update=True, vocabtoks_train, vocabtags_train)
    """

    istream = open(filename, encoding="utf-8")
    corpus = []
    toks, mwes, deprels = [], [], []
    for line in istream:
        line = line.strip()
        if line and line[0] != "#":
            try:
                tokidx, token, lemma, pos, upos, features, headidx, deprel, extended, _ = line.split()

            except ValueError:
                pass

            """
            if tokidx == "1":
                # beginning of sentence, add false toks
                toks.append("<bos>")
                mwes.append("<bos>")
                deprels.append("<bos>")
            """

            # extract simple mwe tags
            extr_mwe = lambda x: "I" if features.startswith("component") else "B"

            mwe = extr_mwe(features)+"_"+upos2pos(pos)
            toks.append(token)
            mwes.append(mwe)
            deprels.append(deprel)

            if update:
                toks_vocab.update(token)
                tags_vocab.update(mwe)
                deprel_vocab.update(deprel)

        elif toks:
            # end of sentence, add  false tokens
            #corpus.append({"tok": toks+ ["<eos>"], "mwe":mwes+["<eos>"], "deprel": deprels+ ["<eos>"] })
            corpus.append({"tok": toks , "mwe": mwes , "deprel": deprels })
            toks, mwes, deprels = [], [], []

    istream.close()
    # return the encoded data in list of list, the nested list represents the sentences
    return corpus, toks_vocab, tags_vocab, deprel_vocab


# [{"token1": "token", "multiword": "mwe", "mwe lemma": "mwe lemma"}, {"token2": "token", "multiword": "mwe"}, {"token3": "token", "multiword": "mwe"}]

class MweRnnDataset(Dataset):

    def __init__(self, datafilename=None, toks_vocab=Vocabulary([ "<unk>", "<pad>"]),
                 tags_vocab=Vocabulary([ "<unk>","<pad>"]),  deprel_vocab=Vocabulary([ "<unk>", "<pad>"]), isTrain=False):
        """
        take as input either the path to a conllu file or a list of tokens
        we consider context size as the n preceding and n subsequent words in the text as the context for predicting the next word.
        """
        super(MweRnnDataset, self).__init__()

        self.toks_vocab, self.tags_vocab,self.deprel_vocab  = toks_vocab, tags_vocab, deprel_vocab
        if datafilename:
            self.corpus, self.toks_vocab, self.tags_vocab, self.deprel_vocab = readfile(datafilename,
                                                                                update=isTrain,
                                                                                toks_vocab=toks_vocab,
                                                                                tags_vocab=tags_vocab)
            self.data = self.corpus

        print('token Vocab size', len(self.toks_vocab))

    def __len__(self):
        return len(self.data)
    @property
    def tag_padidx(self):
        return self.tags_vocab.lookup("<pad>")

    def __getitem__(self, idx):

        x = self.data[idx]["tok"]
        mwe = self.data[idx]["mwe"]
        deprel= self.data[idx]["deprel"]

        return x, mwe, deprel

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
            depl     = []
            max_seq_len = max(len(sent) for sent,_, _ in selected_items)+1

            for sent, mwe, deprel in selected_items:

                x = torch.tensor([self.toks_vocab[tok] for tok in sent])

                y = torch.tensor([self.tags_vocab[tag] for tag in mwe])
                deprel = torch.tensor([self.tags_vocab[tag] for tag in deprel])

                XtoksIDs.append(pad(x, (0, max_seq_len-len(x)), value = self.toks_vocab["<pad>"]))
                YtagsIDs.append(pad(y, (0, max_seq_len-len(y)), value = self.tags_vocab["<pad>"]))
                depl.append(pad(deprel, (0, max_seq_len-len(deprel)), value = self.deprel_vocab ["<pad>"]))
            return torch.stack(XtoksIDs), torch.stack(depl) , torch.stack(YtagsIDs)

        # specify that the mk_batch function should be used to collate the individual samples into batches.
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=mk_batch)

class Mysubset(MweRnnDataset):

    """
    an auxiliary class to take care of the K fold validation, split the train corpus into train and dev
    """
    def __init__(self, subset, toks_vocab, tags_vocab,deprel_vocab):
        self.subset = subset
        self.toks_vocab = toks_vocab
        self.tags_vocab = tags_vocab
        self.deprel_vocab = deprel_vocab

    def __getitem__(self, index):
        return self.subset[index]
    def __len__(self):
        return len(self.subset)

if __name__ == '__main__':
    corpus, toks_vocab, tags_vocab, deprel_vocab = readfile("corpus/train.conllu")