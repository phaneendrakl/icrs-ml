# code for classification :crop_recommendation
import pandas as pd
import os
from django.conf import settings

path = os.path.join(settings.MEDIA_ROOT, 'crop_recommendation.csv')
df = pd.read_csv(path)
# x=df.iloc[0:,0:8].values_counts

# for changing all uppercase into lowercase
def change_case(i):
    i.replace(" ", "")
    i.lower()
    return i


df['label'] = df['label'].apply(change_case)

df['label'] = df['label'].replace('kidneybeans', 'kidneybean')
df['label'] = df['label'].replace('pigeonpeas', 'pigeonpea')
df['label'] = df['label'].replace('mothbeans', 'mothbean')
df['label'] = df['label'].replace('grapes', 'grape')

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'CEC']]
target = df['label']

# training and testing data
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.20, random_state=2)

acc = []
model = []


# DecisionTreeClassifier with cross_validation_score
def build_decisiontree_model():
    from sklearn.tree import DecisionTreeClassifier
    decisiontree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
    decisiontree.fit(xtrain, ytrain)
    ypred = decisiontree.predict(xtest)
    # performance metrics
    from sklearn import metrics
    accuracy = metrics.accuracy_score(ytest, ypred)
    precission = metrics.precision_score(ytest, ypred, average='weighted', zero_division=1)
    f1_score = metrics.f1_score(ytest, ypred, average='weighted')
    recall = metrics.recall_score(ytest, ypred, average='weighted')
    acc.append(accuracy)
    model.append(decisiontree)
    from sklearn.model_selection import cross_val_score
    cscore = cross_val_score(decisiontree, features, target, cv=5)
    print("DecisionTreeClassifier Accuracy: ", accuracy, precission, f1_score, recall)
    return accuracy, precission, f1_score, recall


# Naive_Bayes GuassianNB
def build_naive_model():
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(xtrain, ytrain)
    ypred = nb.predict(xtest)
    # performance Metrics
    from sklearn import metrics
    accuracy = metrics.accuracy_score(ytest, ypred)
    precission = metrics.precision_score(ytest, ypred, average='weighted')
    f1_score = metrics.f1_score(ytest, ypred, average='weighted')
    recall = metrics.recall_score(ytest, ypred, average='weighted')
    acc.append(accuracy)
    model.append('NaiveBayes')
    # Cross_Validation
    from sklearn.model_selection import cross_val_score
    cscore = cross_val_score(nb, features, target, cv=6, scoring='accuracy')
    print("Naive_Bayes: ", accuracy, precission, f1_score, recall, cscore)
    return accuracy, precission, f1_score, recall


def build_svm_model():
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler
    norm = MinMaxScaler().fit(xtrain)
    x_train_norm = norm.transform(xtrain)
    x_test_norm = norm.transform(xtest)
    svm = SVC(kernel='poly', degree=3)
    svm.fit(x_train_norm, ytrain)
    ypred = svm.predict(x_test_norm)
    # Performance Metrics
    from sklearn import metrics
    accuracy = metrics.accuracy_score(ytest, ypred)
    precission = metrics.precision_score(ypred, ytest, average='weighted')
    f1_score = metrics.f1_score(ypred, ytest, average='weighted')
    recall = metrics.recall_score(ypred, ytest, average='weighted')
    acc.append(accuracy)
    model.append('SVM')
    from sklearn.model_selection import cross_val_score
    cscores = cross_val_score(svm, features, target, cv=6, scoring='accuracy')
    print("SVM Results: ", accuracy, precission, f1_score, recall, cscores)
    return accuracy, precission, f1_score, recall


def build_lregression_model():
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(random_state=2)
    lg.fit(xtrain, ytrain)
    ypred = lg.predict(xtest)
    from sklearn import metrics
    accuracy = metrics.accuracy_score(ytest, ypred)
    precission = metrics.precision_score(ytest, ypred, average='weighted')
    f1_score = metrics.f1_score(ypred, ytest, average='weighted')
    recall = metrics.recall_score(ypred, ytest, average='weighted')
    acc.append(accuracy)
    model.append('LogisticRegressor')
    from sklearn.model_selection import cross_val_score
    cscores = cross_val_score(lg, features, target, cv=6, scoring='accuracy')
    print("LogisticRegressor Results: ", accuracy, precission, f1_score, recall, cscores)
    return accuracy, precission, f1_score, recall


def build_random_model():
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=20, random_state=0)
    rfc.fit(xtrain, ytrain)
    ypred = rfc.predict(xtest)
    from sklearn import metrics
    accuracy = metrics.accuracy_score(ytest, ypred)
    precission = metrics.precision_score(ytest, ypred, average='weighted')
    f1_score = metrics.f1_score(ypred, ytest, average='weighted')
    recall = metrics.recall_score(ypred, ytest, average='weighted')
    acc.append(accuracy)
    model.append('RandomForest')
    from sklearn.model_selection import cross_val_score
    cscores = cross_val_score(rfc, features, target, cv=6, scoring='accuracy')
    print("LogisticRegressor Results: ", accuracy, precission, f1_score, recall, cscores)
    return accuracy, precission, f1_score, recall


# def build_xgboost_model():
#     import xgboost as xgb
#     xb = xgb.XGBClassifier(use_label_encoder=False)
#     xb.fit(xtrain, ytrain)
#     ypred = xb.predict(xtest)
#     from sklearn import metrics
#     accuracy = metrics.accuracy_score(ytest, ypred)
#     precission = metrics.precision_score(ytest, ypred, average='weighted')
#     f1_score = metrics.f1_score(ypred, ytest, average='weighted')
#     recall = metrics.recall_score(ypred, ytest, average='weighted')
#     acc.append(accuracy)
#     model.append('XGBoost')
#     from sklearn.model_selection import cross_val_score
#     cscores = cross_val_score(xb, features, target, cv=6, scoring='accuracy')
#     print("XGBoostClassifier Results: ", accuracy, precission, f1_score, recall, cscores)
#     return accuracy, precission, f1_score, recall

def build_neuralnetwork_model():
    from sklearn.neural_network import MLPClassifier
    # from sklearn.preprocessing import MinMaxScaler
    # norm=MinMaxScaler().fit(xtrain)
    # x_train_norm=norm.transform(xtrain)
    # x_test_norm=norm.transform(xtest)
    mlpclassifier = MLPClassifier(random_state=2, max_iter=550)
    mlpclassifier.fit(xtrain, ytrain)
    ypred = mlpclassifier.predict(xtest)
    from sklearn import metrics
    accuracy = metrics.accuracy_score(ytest, ypred)
    precission = metrics.precision_score(ytest, ypred, average='weighted')
    f1_score = metrics.f1_score(ypred, ytest, average='weighted')
    recall = metrics.recall_score(ypred, ytest, average='weighted')
    acc.append(accuracy)
    model.append('NeuralNetwork')
    from sklearn.model_selection import cross_val_score
    cscores = cross_val_score(mlpclassifier, features, target, cv=6, scoring='accuracy')
    print("NeuralNetwork Results: ", accuracy, precission, f1_score, recall, cscores)
    return accuracy, precission, f1_score, recall


def plotting():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=[10, 6], dpi=100)
    plt.title('Accuracy Comparision Chart')
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    pf = sns.barplot(x=['DecisionTree', 'NAIVEBAYES', 'SVM', 'LogReg', 'RanFor', 'NeuralNetwork'],
                     y=[90.454, 99.090, 97.727, 95.681, 98.863, 95.681],
                     palette='dark')
    plt.show()
    return pf


def build_neuralnetwork_model2(data):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    # norm=MinMaxScaler().fit(xtrain)
    # x_train_norm=norm.transform(xtrain)
    # x_test_norm=norm.transform(xtest)
    p = os.path.join(settings.MEDIA_ROOT, 'crop_recommendation.csv')
    f = pd.read_csv(p)
    x = f[['N', 'P', 'K', 'ph', 'rainfall']]
    y = f['label']
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    mlpclass = MLPClassifier(random_state=2, max_iter=550)
    mlpclass.fit(x_train, y_train)
    result = mlpclass.predict(data)
    return f"You can Grow {result} in your fields"
