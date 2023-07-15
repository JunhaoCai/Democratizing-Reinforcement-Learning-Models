"""
配置参数
"""
class Config():
    update_w2v = True          # 是否在训练中更新w2v
    vocab_size = 54848          # 词汇量，与word2id中的词汇量一致
    n_class = 2                 # 分类数：分别为pos和neg
    max_sen_len = 65           # 句子最大长度
    embedding_dim = 50          # 词向量维度

    train_path = '../word2vec_data/train.txt'
    val_path = '../word2vec_data/validation.txt'
    test_path = '../word2vec_data/test.txt'
    pre_path = '../word2vec_data/pre.txt'
    word2id_path = '../word2vec_data/word2id.txt'
    pre_word2vec_path = '../word2vec_data/wiki_word2vec_50.bin'
    corpus_word2vec_path = '../word2vec_data/word_vec.txt'
    model_state_dict_path='./word2vec_data/SVM_model/' # 训练模型保存的地址