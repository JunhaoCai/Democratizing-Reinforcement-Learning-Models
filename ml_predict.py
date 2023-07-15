import numpy as np
import gensim
import joblib
import jieba


# 加载预训练的词向量模型
# model = gensim.models.KeyedVectors.load_word2vec_format("word2vec_data/wiki_word2vec_50.bin", binary=True)
# print(model)

def preprocess_text(text):
    # 进行文本清洗和标准化，去除特殊字符等
    # 示例：去除标点符号
    processed_text = text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
    return processed_text

def extend_word_vector(word_vector):
    extended_vector = np.append(word_vector, np.zeros(15))
    return extended_vector

def extract_features(text, model):
    # 提取特征表示
    words = jieba.lcut(text, cut_all=True)
    features = []
    for word in words:
        if word in model.key_to_index:
            word_vector = model.get_vector(word)
            if word_vector.shape[0] == 50:
                word_vector = extend_word_vector(word_vector)
            features.append(word_vector)
    if len(features) > 0:
        features = np.mean(features, axis=0)  # 使用平均词向量作为特征
    else:
        features = np.zeros(65)  # 如果没有有效词向量，则使用全零向量
    return features

def predict(query, ml_model, model):
    processed_query = preprocess_text(query)
    features = extract_features(processed_query, model)
    features = features.reshape(1, -1)
    # print("ML_predict_features: ", features)
    result = ml_model.predict(features)[0]

    if result == 0:
        # print('类别：好评')
        return 0

    elif result == 1:
        # print('类别：坏评')
        return 1