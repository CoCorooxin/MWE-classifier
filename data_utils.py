import json
class Vocabulary:
    """
    general encoding class mapping str -> index and vice versa
    """

    def __init__(self, symbols=None):

        # dictionary to map the vocabulary with a id to build matrix
        # add "UNK" for unknown word to the initial mapping
        self.word2idx = dict()
        self.idx2word = []

        if symbols:
            for sym in symbols:
                self.update(sym)

    @staticmethod
    def read(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            toks = f.read().split(" ")
        return Vocabulary(toks)

    def write(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(" ".join(self.idx2word))

    def update(self, tok):

        # takes as input a symbol and build the mapping if it doesnt exist
        if tok not in self.word2idx:
            self.word2idx[tok] = len(self.idx2word)
            self.idx2word.append(tok)

    def lookup(self, tok, update=False):

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

    def __call__(self, tok):
        return self.lookup(tok)


def save_word_embeddings(embeddings, output_file):
    embedding_dict = {}
    for token, embedding in embeddings.items():
        embedding_dict[token] = embedding.tolist()

    with open(output_file, 'w') as json_file:
        json.dump(embedding_dict, json_file)
