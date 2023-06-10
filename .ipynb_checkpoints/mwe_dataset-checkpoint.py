from torch.utils.data import Dataset, DataLoader
import torch



class Vocabulary:
    
    def __init__(self, symbols = None):
        
        #dictionary to map the vocabulary with a id to build matrix
        #add "UNK" for unknown word to the initial mapping  
        self.word2idx = dict()
        self.idx2word = []

        if symbols:
            for sym in symbols:
                self.update(sym)
    @staticmethod
    def read(self, vocab_file):
        with open(vocab_file, "r", encoding = "utf-8") as f:
            toks = f.read().split(" ")
        return Vocabulary(toks)

    def write(self, filename):
        with open(filename, "w", encoding = "utf-8") as f:
            f.write(" ".join(self.idx2word))

    def update(self, tok):
        
        #takes as input a symbol and build the mapping if it doesnt exist
        if tok not in self.word2idx:
            self.word2idx[tok] = len(self.idx2word)
            self.idx2word.append(tok)

    def lookup(self, tok, update = False):
        
        #find tok id given the string, if the tok does not exist return the idx of "UNK"
        if tok not in self.word2idx:
            if update:
                self.update(tok)
                return self[tok]
            return self.word2idx["<unk>"]
            
        return self.word2idx[tok]
    def rev_lookup(self, idx):
        
        #find the tok string given the id
        return self.idx2word[idx]
    
    def __getitem__(self, symbol):
        
        #if the symbol does not exist we see it as unk
        return self.lookup(symbol)
    
    def __len__(self):
        
        return len(self.idx2word)
    
"""
Functions for reading and writing UD CONLL data
"""
CONLL_FIELDS = ["token", "pos", "features", "deprel"]
MWE_TAGS = ["B", "I"]  # B for begin , I for inside


def readfile(filename, update=False, toks_vocab=Vocabulary(["<unk>", "<bos>", "<eos>"]),
             tags_vocab=Vocabulary(["B_X"])):
    """
    function to read and encode the corpus at one pass
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
                tokidx, token, lemma, upos, pos, features, headidx, deprel, extended, _ = line.split()

            except ValueError:
                pass
            if tokidx == "1":
                # beginning of sentence, add false toks
                sent_toks.append(toks_vocab["<bos>"])
                sent_tags.append(tags_vocab["B_X"])

            # extract simple mwe tags
            mwe_tag = lambda x: "I" if features.startswith("component") else "B"
            # extract tagging information
            sent_toks.append(toks_vocab.lookup(tok=token, update=update))
            sent_tags.append(tags_vocab.lookup(tok=mwe_tag(features) + "_" + upos, update=update))

        elif sent_toks:
            # end of sentence, add  false tokens
            sent_toks.append(toks_vocab["<eos>"])
            sent_tags.append(tags_vocab["B_X"])
            X_toks.append(sent_toks)
            Y_tags.append(sent_tags)
            sent_toks, sent_tags = [], []

    istream.close()
    # return the encoded data in list of list, the nested list represents the sentences
    return X_toks, Y_tags, toks_vocab, tags_vocab


# [{"token1": "token", "multiword": "mwe", "mwe lemma": "mwe lemma"}, {"token2": "token", "multiword": "mwe"}, {"token3": "token", "multiword": "mwe"}]


class MWEDataset(Dataset):

    def __init__(self, datafilename=None, toks_vocab=Vocabulary(["<unk>", "<bos>", "<eos>"]),
                 tags_vocab=Vocabulary(["B_X"]), isTrain=False, window_size=0):
        """
        take as input either the path to a conllu file or a list of tokens
        we consider context size as the n preceding and n subsequent words in the text as the context for predicting the next word.
        """
        super(MWEDataset, self).__init__()

        self.toks_vocab, self.tags_vocab = toks_vocab, tags_vocab

        self.Xtoks_IDs, self.Ytags_IDs, self.toks_vocab, self.tags_vocab = readfile("corpus/train.conllu",
                                                                                    update=isTrain,
                                                                                    toks_vocab=toks_vocab,
                                                                                    tags_vocab=tags_vocab)

        print('token Vocab size', len(self.toks_vocab))
        self.window_size = window_size
        self.data = self.build_dataset(self.Xtoks_IDs, self.Ytags_IDs)

    def __len__(self):
        return len(self.data)

    def build_dataset(self, X_toks, Y_tags):
        """
        build examples with contextual tokens as features
        takes as input a nested list of encoded corpus, [sentences[tokens]]
        return a list of examples with context window features
        """
        examples = []
        for toks, tags in zip(X_toks, Y_tags):

            toks = [self.toks_vocab["<bos>"]] * self.window_size + toks + [
                self.toks_vocab["<eos>"]] * self.window_size  # 3+3+3

            for i in range(self.window_size, len(toks) - self.window_size, 1):  # 3, 6, 1

                examples.append((torch.tensor(toks[i - self.window_size: i + self.window_size + 1]),
                                 torch.tensor(tags[i - self.window_size])))
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
            out.append([self.tok_vocab.rev_lookup(idx) for idx in line])
        return out

    def get_loader(self, batch_size=1, num_workers=0, word_dropout=0., shuffle=False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


