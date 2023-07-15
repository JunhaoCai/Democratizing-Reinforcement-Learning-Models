from __future__ import unicode_literals, print_function, division
from io import open
import torch
import re
import numpy as np
import gensim
from torch.utils.data import Dataset
from nn_Config import Config


class Data_set(Dataset):
    """
    自定义数据类，只需要定义__len__和__getitem__这两个方法就可以。
    我们可以通过迭代的方式来取得每一个数据，但是这样很难实现取batch，shuffle或者多线程读取数据，此时，需要torch.utils.data.DataLoader来进行加载
    """
    def __init__(self, Data, Label):
        self.Data = Data
        # 考虑对测试集的使用
        if Label is not None:
            self.Label = Label

    def __len__(self):
        # 返回长度
        return len(self.Data)

    def __getitem__(self, index):
        # 如果是训练集
        if self.Label is not None:
            data = torch.from_numpy(self.Data[index])
            label = torch.from_numpy(self.Label[index])
            return data, label
        # 如果是测试集
        else:
            data = torch.from_numpy(self.Data[index])
            return data


def stopwordslist():
    """
    创建停用词表
    :return:
    """
    stopwords = [line.strip() for line in open('../word2vec_data/stopword.txt', encoding='UTF-8').readlines()]
    return stopwords


def build_word2id(file):
    """
    将word2id词典写入文件中，key为word，value为索引
    :param file: word2id保存地址
    :return: None
    """
    # 加载停用词表
    stopwords = stopwordslist()
    word2id = {'_PAD_': 0}
    # 文件路径
    path = [Config.train_path, Config.val_path]
    # print(path)
    # 遍历训练集与验证集
    for _path in path:
        # 打开文件
        with open(_path, encoding='utf-8') as f:
            # 遍历文件每一行
            for line in f.readlines():
                out_list = []
                # 去掉首尾空格并按照空格分割
                sp = line.strip().split()
                # 遍历文本部分每一个词
                for word in sp[1:]:
                    # 如果词不是停用词
                    if word not in stopwords:
                        # 在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。
                        rt = re.findall('[a-zA-Z]+', word)
                        # 如果word不等于制表符
                        if word != '\t':
                            # 如果词匹配的字符串为1，则继续遍历下一个词
                            if len(rt) == 1:
                                continue
                            # 如果词匹配的字符串为0，则将这个词添加到out_list中
                            else:
                                out_list.append(word)

                # 遍历out_list中的词
                for word in out_list:
                    # 如果这些词不在word2id字典的key中,则添加到word2id字典中
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    # 打开输出文件并进行文件写入
    with open(file, 'w', encoding='utf-8') as f:
        # 遍历词典中的每一个词
        for w in word2id:
            f.write(w + '\t')
            f.write(str(word2id[w]))
            f.write('\n')


def build_word2vec(fname, word2id, save_to_path=None):
    """
    使用word2vec对单词进行编码
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    # 词的总数量
    n_words = max(word2id.values()) + 1
    # 加载预训练的词向量
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    # 初始化词向量
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    # 遍历每个单词
    for word in word2id.keys():
        try:
            # 构建词向量
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    # 将word_vecs保存到文件中
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    # 返回word_vecs数组
    return word_vecs


def text_to_array(word2id, seq_lenth, path):
    """
    有标签文本转为索引数字模式
    :param word2id: word2id
    :param seq_lenth: 句子最大长度
    :param path: 文件路径
    :return:
    """
    # 存储标签
    lable_array = []
    # 句子索引初始化
    i = 0
    sa = []

    # 获取句子个数
    with open(path, encoding='utf-8') as f1:
        # 打开文件并遍历文件每一行
        for l1 in f1.readlines():
            # 返回分割后的字符串列表
            s = l1.strip().split()
            # 去掉标签
            s1 = s[1:]
            # 单词转索引数字
            new_s = [word2id.get(word, 0) for word in s1]
            # 存储由索引数字表示的文本列表
            sa.append(new_s)
        # print(len(sa))

    with open(path, encoding='utf-8') as f:
        # 初始化句子array；行：句子个数 列：句子长度
        sentences_array = np.zeros(shape=(len(sa), seq_lenth))
        # 遍历每一句话
        for line in f.readlines():
            # 返回分割后的字符串列表
            sl1 = line.strip().split()
            # 去掉标签
            sen = sl1[1:]
            # 单词转索引数字,不存在则为0
            new_sen = [word2id.get(word, 0) for word in sen]
            # 转换为(1,sen_len)
            new_sen_np = np.array(new_sen).reshape(1, -1)

            # 补齐每个句子长度，少了就直接赋值,0填在前面。
            # np.size，返回沿给定轴的元素数
            if np.size(new_sen_np, 1) < seq_lenth:
                sentences_array[i, seq_lenth - np.size(new_sen_np, 1):] = new_sen_np[0, :]
            # 长了进行截断
            else:
                sentences_array[i, 0:seq_lenth] = new_sen_np[0, 0:seq_lenth]

            i = i + 1
            # 标签
            lable = int(sl1[0])
            lable_array.append(lable)
    # 返回索引模式的文本以及标签
    return np.array(sentences_array), lable_array


def text_to_array_nolable(word2id, seq_lenth, path):
    """
    无标签文本转为索引数字模式,与上面相比，只是少了标签的处理
    :param word2id:
    :param seq_lenth: 序列长度
    :param path:文件路径
    :return:
    """

    i = 0
    sa = []
    # 获取句子个数
    with open(path, encoding='utf-8') as f1:
        # 打开文件并遍历文件每一行
        for l1 in f1.readlines():
            # 返回分割后的字符串列表
            s = l1.strip().split()
            # 去掉标签
            s1 = s[1:]
            # 单词转索引数字
            new_s = [word2id.get(word, 0) for word in s1]
            # 存储由索引数字表示的文本列表
            sa.append(new_s)


    with open(path, encoding='utf-8') as f:
        # 初始化句子array；行：句子个数 列：句子长度
        sentences_array = np.zeros(shape=(len(sa), seq_lenth))
        # 遍历每一句话
        for line in f.readlines():
            # 返回分割后的字符串列表
            sl1 = line.strip().split()
            # 去掉标签
            sen = sl1[1:]
            # 单词转索引数字,不存在则为0
            new_sen = [word2id.get(word, 0) for word in sen]
            # 转换为(1,sen_len)
            new_sen_np = np.array(new_sen).reshape(1, -1)

            # 补齐每个句子长度，少了就直接赋值,0填在前面。
            # np.size，返回沿给定轴的元素数
            if np.size(new_sen_np, 1) < seq_lenth:
                sentences_array[i, seq_lenth - np.size(new_sen_np, 1):] = new_sen_np[0, :]
            # 长了进行截断
            else:
                sentences_array[i, 0:seq_lenth] = new_sen_np[0, 0:seq_lenth]
            i = i + 1
    # 返回索引模式的文本
    return np.array(sentences_array)


def to_categorical(y, num_classes=None):
    """
    将类别转化为one-hot编码
    :param y: 标签
    :param num_classes: 类别数
    :return:
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    # 压平
    y = y.ravel()
    # 计算类别数
    if not num_classes:
        num_classes = np.max(y) + 1

    n = y.shape[0]
    # 初始化
    categorical = np.zeros((n, num_classes))
    # 赋值
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def prepare_data(w2id, train_path, val_path, test_path, seq_lenth):
    """
    得到数字索引表示的句子和标签
    :param w2id: word2id
    :param train_path: 训练文件路径
    :param val_path: 验证文件路径
    :param test_path: 测试文件路径
    :param seq_lenth: 句子最大长度
    :return:
    """
    # 对训练集、验证集、测试集处理，将文本转化为由单词索引构成的array
    train_array, train_lable = text_to_array(w2id, seq_lenth=seq_lenth, path=train_path)
    val_array, val_lable = text_to_array(w2id, seq_lenth=seq_lenth, path=val_path)
    test_array, test_lable = text_to_array(w2id, seq_lenth=seq_lenth, path=test_path)

    # 标签为[1, 1, 1, 1, 1, 1, 1, 1, 0, 0...]将标签转为onehot
    # train_lable=to_categorical(train_lable,num_classes=2)
    # val_lable=to_categorical(val_lable,num_classes=2)

    """for i in train_lable:
        np.array([i])"""
    # 转换标签数据格式
    train_lable = np.array([train_lable]).T
    val_lable = np.array([val_lable]).T
    test_lable = np.array([test_lable]).T
    """转换后标签
            [[0. 1.]
            [0. 1.]
            [0. 1.]
            ...
            [1. 0.]
            [1. 0.]
            [1. 0.]]"""
    # print(train_lab,"\nval\n",val_lab)
    # 返回训练集、验证集、测试集的array与label
    return train_array, train_lable, val_array, val_lable, test_array, test_lable

if __name__ == '__main__':
    # 建立word2id，并将word2id写入文件中
    build_word2id('../word2vec_data/word2id.txt')
    splist = []
    # 基于文件重新构建word2id，这里也可以将build_word2id中的word2id返回
    word2id = {}
    with open('../word2vec_data/word2id.txt', encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()  # 去掉\n \t 等
            splist.append(sp)
        word2id = dict(splist)  # 转成字典

    # 将word2id中的value转化为int
    for key in word2id:
        word2id[key] = int(word2id[key])

    # 构建id2word
    id2word = {}
    for key, val in word2id.items():
        id2word[val] = key

    # 构建word2vec词向量
    w2vec = build_word2vec(Config.pre_word2vec_path, word2id, Config.corpus_word2vec_path)

    # 得到句子id表示和标签
    train_array, train_lable, val_array, val_lable, test_array, test_label = prepare_data(word2id,
                                                                                          train_path=Config.train_path,
                                                                                          val_path=Config.val_path,
                                                                                          test_path=Config.test_path,
                                                                                          seq_lenth=Config.max_sen_len)
    # 将训练集、验证集、测试集处理后的句子id表示保存至文件中
    np.savetxt('./word2vec_data/train_data.txt', train_array, fmt='%d')
    np.savetxt('./word2vec_data/val_data.txt', val_array, fmt='%d')
    np.savetxt('./word2vec_data/test_data.txt', test_array, fmt='%d')
