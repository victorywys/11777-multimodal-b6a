class global_consts():
    debug = False

    cuda = 0
    device = None

    wv_path = "../../data/glove/glove.6B.300d.txt"

    PAD_id = 0
    BOS_id = 1
    UNK_id = 2
    EOS_id = 3
    batch_size = 32
    epoch_num = 30
    max_len = 25
    learning_rate = 5e-4

    input_dim = 0
    word_dim = 300
    cell_dim = 512
    vocab_size = 0
