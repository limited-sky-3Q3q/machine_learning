import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import os
import pickle
import cv2 as cv
import random
from feature import NPDFeature
from ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


valid_precision_list = []
iterations_list=[]

facePath = 'datasets\\original\\face\\'
nonfacePath = 'datasets\\original\\nonface\\'

face_data_file = 'datasets\\face_data.txt'
nonface_data_file = 'datasets\\nonface_data.txt'



def dumpFeatures(path, label = 1):  #缓存特征数据
    if label == 1:
        file = face_data_file
    else:
        file = nonface_data_file
    write_feature = open(file, 'wb')
    feature_cache = []  #缓存特征
    
    img_names = os.listdir(path) #返回指定的文件夹包含的图片的名字的列表
    for name in img_names:
        gray_img = cv.resize(cv.imread(path+name, cv.IMREAD_GRAYSCALE), (24,24))   #转化大小为24*24的灰度图
        npd = NPDFeature(gray_img)
        feature_cache.append(npd.extract())
    pickle.dump(feature_cache, write_feature)

def load_the_features(filename):  #将缓存的特征读入
    read_feature = open(filename, 'rb')
    data = pickle.load(file=read_feature)
    return np.matrix(data)  #构建matrix对象

def add_the_label(data, label):
    data = np.matrix(data)
    return np.concatenate((np.full(shape=(data.shape[0],1),fill_value=label), data), axis=1)

def split_train_valid(data, fraction = 0.9):
    data = np.matrix(data)
    return sk.model_selection.train_test_split(data[:,0],data[:,1:data.shape[1]],train_size=fraction, test_size=1-fraction)

def draw_image(iterations, valid_precision):
    plt.figure(figsize=(8,6))  # 定义图的大小
    plt.xlabel("iteration")     # X轴标签
    plt.ylabel(" valid_precision")        # Y轴坐标标签
    plt.title("")      #  曲线图的标题
    plt.plot(iterations, valid_precision)            # 绘制曲线图
    #在ipython的交互环境中需要这句话才能显示出来
    plt.show()

if __name__=="__main__":
    #缓存特征数据
    dumpFeatures(facePath, label=1)
    dumpFeatures(nonfacePath, label=0)

    #载入数据
    face_data = load_the_features(face_data_file)
    nonface_data = load_the_features(nonface_data_file)

    face_data = add_the_label(face_data, 1)   #给图片添加标签, 人脸的标签是1, 非人脸是0
    nonface_data = add_the_label(nonface_data, 0)
    #print("1")

    all_data = np.concatenate((face_data, nonface_data), axis=0) #数据拼接


    #划分数据集, 训练集占90%
    X_train, X_valid, y_train, y_valid = sk.model_selection.train_test_split(all_data[:, 1:all_data.shape[1]], all_data[:, 0], train_size=0.9, test_size=0.1)

    #构建matrix对象
    X_train = np.matrix(X_train)
    X_valid = np.matrix(X_valid)
    y_train = np.matrix(y_train)
    y_valid = np.matrix(y_valid)


    #print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    print(np.sum(y_train==1), np.sum(y_valid==1))


    #weight = np.full(shape=(X_train.shape[0]), fill_value=1/X_train.shape[0])   #生成数组

    weak_classifier = DecisionTreeClassifier()  #自动生成弱分类器, 但其实是在AdaBoostClassifier中生成, 这里的没用到
    for i in range(0,11):
        classifier = AdaBoostClassifier(weak_classifier, i)  #生成10个弱分类器, 迭代10次
        #print("9")

        classifier.fit(X_train, y_train)
        # hx = classifier.predict(X_valid)
        train_score = classifier.predict_scores(X_train, y_train)
        valid_score = classifier.predict_scores(X_valid, y_valid)
        #print("10")
        print("迭代次数是 "+str(i)+" 时: ")
        print('训练集分数是 {}, 验证集分数是 {}'.format(train_score, valid_score))

        hx = classifier.predict(X_valid)  #预测验证集的结果
        #print("11")
        report_file = 'report.txt'
        file_write = open(report_file, 'w')
        target_names=['nonface', 'face']
        file_write.write(classification_report(y_valid, hx,target_names=target_names))  #将正确的数据标签和预测的数据标签进行对比,得到准确率
        print(classification_report(y_valid, hx))
        iterations_list.append(i)
        valid_precision_list.append(valid_score)
    draw_image(iterations_list, valid_precision_list)




