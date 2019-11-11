class global_consts():
    debug = False
    inference = True
    checkpoint_id = 12

    cuda = 0
    device = None

    refer_data_path = "../data/"
    image_path = "../../mscoco_feature/trainval"
    box_path = "../../mscoco_feature/boxes/bu_feature_gt/refcoco_unc"
    checkpoint_path = "./checkpoint"
    dataset = "refcoco"
    split_by = "unc"
    wv_path = "/data/glove_vector/glove.6B.300d.txt"

    PAD_id = 0
    BOS_id = 1
    UNK_id = 2
    EOS_id = 3

    min_occur = 0

    batch_size = 32
    epoch_num = 30
    max_len = 10
    learning_rate = 5e-7

    input_padding = 0
    output_padding = 0

    input_dim = 0
    reduce_dim = 1024
    word_dim = 300
    cell_dim = 1024
    vocab_size = 0
