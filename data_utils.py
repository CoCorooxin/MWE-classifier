class Vocabulary:

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


def upos2pos(ud_pos_tag):
    """
    Convert a UPOS tag to the simple POS tag scheme.
    """
    # Define a mapping from UD POS tags to the other scheme
    mapping = {
        'NOUN': 'N',
        'VERB': 'V',
        'ADJ': 'A',
        'ADV': 'ADV',
        'PRON': 'CL',
        'PRO': 'CL',
        'DET': 'D',
        'ADP': 'P',
        'PUNCT': 'PONCT',
        'CCONJ': 'C',
        'SCONJ': 'C',
        'NUM': 'NC',
        'PART': 'P',
        'INTJ': 'I',
        'X': 'X',
        'SYM': 'S',
        'CONJ': 'C',
        "PROPN": 'N',
        'AUX': 'V',
    }

    # Map the UD POS tag to the other scheme, or use a default value if not found
    other_pos_tag = mapping.get(ud_pos_tag, ud_pos_tag)

    return other_pos_tag