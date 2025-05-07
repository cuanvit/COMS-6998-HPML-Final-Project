class GPTConfig:
    def __init__(self, vocab_size, max_len, 
                 n_layer=6, n_head=8, n_embd=512,
                 attn_dropout=0.1, embed_dropout=0.1, ff_dropout=0.1):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_dropout = attn_dropout
        self.embed_dropout = embed_dropout
        self.ff_dropout = ff_dropout