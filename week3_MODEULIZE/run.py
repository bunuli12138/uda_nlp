# -*- coding: utf-8 -*-

"""
Target: pridict whether a English mail is spam or not
"""
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import pickle

def getModel(path='./model/naiveBayes_spamCla_0621_9874.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def getProcessedText(text):
    X_train = pd.read_csv(r'./data/X_train.csv')['sms_message']
    
    if type(X_train) is not pd.core.series.Series:
        raise TypeError('X_train is not a Series')
    
    
    """特征提取"""
    count_vector = CountVectorizer()                        # 初始化

    count_vector.fit_transform(X_train)                               # 训练文本
    
    t_d = count_vector.transform(text)                    # 转换指定文本
    
    return t_d


def getPredResult(model, processedText):
    result = model.predict(processedText)
    
    if result:      return True
    else:           return False


if __name__ == '__main__':
    with open('./text/text.txt', 'r') as f:
        text = f.readlines()
    print(text)
    
    model = getModel()
    
    p_t = getProcessedText(text)
    
    result = getPredResult(model, p_t)
    print(text)
    print('**%s**'%result)
