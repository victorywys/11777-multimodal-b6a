class global_consts():
    debug = False

    cuda = 0
    device = None

    wv_path = "../../data/glove/glove.6B.300d.txt"

    PAD_id = 0
    BOS_id = 1
    UNK_id = 2
    EOS_id = 3
    batch_size = 64
    epoch_num = 200
    padding_len = 10
    max_len = 10
    learning_rate = 1e-5

    cell_type = "LSTM"

    word_dim = 300
    cell_dim = 512
    num_layers = 2

    input_dim = 300

    vocab_size = 0
