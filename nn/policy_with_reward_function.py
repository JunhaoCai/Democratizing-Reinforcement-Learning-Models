import numpy as np
import joblib
import torch
from ml_predict import predict
from nn.nn_predict import classify_review

import gensim

# 加载预训练的词向量模型
model = gensim.models.KeyedVectors.load_word2vec_format("../word2vec_data/wiki_word2vec_50.bin", binary=True)

# 加载训练好的模型参数
svm_model = joblib.load('../SVM/SVM_model.pkl')
naive_bayes_model = joblib.load('../Bayes/NaiveBayes_model.pkl')
decision_tree_model = joblib.load('../DecisionTree/GridSearch_decision_tree_model.pkl')
random_forest_model = joblib.load('../RandomForest/RandomForest_model.pkl')
neural_network_model = torch.load('../nn/nn_models/nn_model_epoch_12.pkl')

# 读取权重文件
def read_weights():
    with open('../weights.txt', 'r') as file:
        weights = file.read().split(',')
        return np.array([float(w) for w in weights])

# 写入权重文件
def write_weights(weights):
    with open('../weights.txt', 'w') as file:
        normalized_weights = weights / np.sum(weights)  # 归一化处理
        file.write(','.join(str(w) for w in normalized_weights))

# 使用模型进行预测
def get_predictions(comment, weights):
    svm_prediction = predict(comment, svm_model, model)
    naive_bayes_prediction = predict(comment, naive_bayes_model, model)
    decision_tree_prediction = predict(comment, decision_tree_model, model)
    random_forest_prediction = predict(comment, random_forest_model, model)
    neural_network_prediction = classify_review(comment, neural_network_model)

    # 根据权重给各个模型的预测结果赋值
    weighted_predictions = [weights[0] * svm_prediction,
                            weights[1] * naive_bayes_prediction,
                            weights[2] * decision_tree_prediction,
                            weights[3] * random_forest_prediction,
                            weights[4] * neural_network_prediction]

    return weighted_predictions

# 更新权重
def update_weights(weights, predictions, true_label, learning_rate, discount_factor):
    rewards = [int(true_label == i) for i in range(len(predictions))]  # 与真实标签比较，若预测准确则奖励为1，否则为0

    # 计算总奖励
    total_reward = sum(rewards)

    for i in range(len(predictions)):
        if rewards[i] == 0:
            # 计算错误预测的相对权重
            error_weight = predictions[i] / (total_reward + 1e-8)

            # 减少预测错误的权重
            weights[i] -= learning_rate * error_weight
        else:
            # 增加预测成功的权重
            weights[i] += learning_rate * (1 - predictions[i])

    # 使用折扣因子进行权重衰减
    weights *= discount_factor

    return weights

weights = read_weights()

learning_rate = 0.1  # 学习率
discount_factor = 0.9  # 折扣因子

while True:
    user_comment = input("请输入你的评论（输入'q'退出）：")
    if user_comment == 'q':
        break

    # 使用模型进行预测
    predictions = get_predictions(user_comment, weights)

    # 输出模型预测类别
    print("模型预测类别：")
    for i, prediction in enumerate(predictions):
        print(f"{i+1}. {prediction}")

    # 根据权重选择最终的判断结果
    final_result = np.argmax(predictions)
    print("最终评论结果：好评" if final_result == 0 else "最终评论结果：坏评")

    # 根据模型预测结果判断是否需要人工干预
    if np.argmax(predictions) != np.argmin(predictions):
        choice = input("请判断模型预测是否正确（输入'y'表示正确，或输入'n'进行人工干预）：")
        if choice.lower() == 'y':
            continue
        elif choice.lower() == 'n':
            correct_label = int(input("请输入正确的评价类别（0表示好评，1表示坏评）："))

            # 更新权重
            weights = update_weights(weights, predictions, correct_label, learning_rate, discount_factor)

            # 将新的权重写入文件
            write_weights(weights)