# -*-coding:utf-8-*-
import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
import numpy as np
import warnings


def feature_normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0, ddof=1)
    return np.nan_to_num((data - mu) / sigma)


def get_features(selected_data):
    s = sum(selected_data)
    x = np.empty(shape=[0, s], dtype=np.float64)
    y = np.empty(shape=[0], dtype=np.float64)

    with open("../dataset/features.txt", encoding="utf-8") as f:
        for line in f:
            content = line.strip().split('\t')
            vec = np.array([[float(content[x]) for x in range(len(content)) if selected_data[x] == 1]], dtype=np.float64)
            x = np.append(x, vec, axis=0)
            y = np.append(y, float(content[-1]))
        f.close()

    return x, y


def get_feature_matrix():
    n_year = 52
    x = np.empty(shape=[0, n_year * 5], dtype=np.float64)
    y = np.empty(shape=[0], dtype=np.float64)

    with open("../dataset/features_matrix.txt", encoding="utf-8") as f:
        cnt = 0
        vec = np.zeros((1, n_year * 5), dtype=np.float64)
        for line in f:
            content = line.strip().split('\t')
            cnt += 1

            if cnt == 1:
                pass

            elif cnt < 6:
                for i in range(n_year):
                    vec[0][i + (cnt - 2) * n_year] = float(content[i])

            elif cnt == 6:
                for i in range(n_year):
                    vec[0][i + (cnt - 2) * n_year] = float(content[i])
                x = np.append(x, vec, axis=0)
                vec = np.zeros((1, n_year * 5), dtype=np.float64)

            elif cnt == 7:
                y = np.append(y, float(content[-1]))
                cnt = 0
        f.close()
    return x, y


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    select_feature = [0,
                      0,
                      1,  # 题目相似度
                      0,  # 学术年龄之差
                      1,  # n1文章数
                      1,  # n2文章数
                      1,  # 文章总数与n1文章数比值
                      1,  # 文章总数与n2文章数比值
                      0,  # 合作次数
                      0,  # 合作时间
                      1,  # 合作论文数与n1论文数比值
                      1,  # 合作论文数与n2论文数比值
                      1,  # 合作之前n1论文数
                      1,  # 合作之前n2论文数
                      0,  # kulc
                      0,  # ir
                      0]
    #x, y = get_feature_matrix()
    #print(x.shape)

    x, y = get_features(select_feature)
    x = feature_normalize(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    times = 1

    # Logistic Regression
    model_LR = LogisticRegression()
    model_LR.fit(x_train, y_train)
    score_LR = 0
    for time in range(times):
        score_LR += cross_val_score(model_LR, x_test, y_test, cv=10, scoring='accuracy').mean()
    print("The score of Logistic Regression is : ",  score_LR / times)

    # SVM kernel = 'rbf'
    clf_rbf = svm.SVC(kernel='rbf')
    clf_rbf.fit(x_train, y_train)
    score_rbf = 0
    for time in range(times):
        score_rbf += cross_val_score(clf_rbf, x_test, y_test, cv=10, scoring='accuracy').mean()
    print("The score of SVM rbf is : ", score_rbf / times)

    # SVM kernel = 'linear'
    clf_linear = svm.SVC(kernel='linear')
    clf_linear.fit(x_train, y_train)
    score_linear = 0
    for time in range(times):
        score_linear += cross_val_score(clf_linear, x_test, y_test, cv=10, scoring='accuracy').mean()
    print("The score of SVM linear is : ", score_linear / times)

    # XGBoost
    model_xgboost = XGBClassifier()
    model_xgboost.fit(x_train, y_train)
    score_xgboost = 0
    for time in range(times):
        score_xgboost += cross_val_score(model_xgboost, x_test, y_test, cv=10, scoring='accuracy').mean()
    print("The score of XGBoost is : ", score_xgboost / times)

    # Random Forest
    model_RF = RandomForestClassifier()
    model_RF.fit(x_train, y_train)
    score_RF = 0
    for time in range(times):
        score_RF += cross_val_score(model_RF, x_test, y_test, cv=10, scoring='accuracy').mean()
    print("The score of Random Forest is : ", score_RF / times)

