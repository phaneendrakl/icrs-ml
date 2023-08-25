# Code for NeuralNetwork(MultiLayerPerceptron)
import pandas as pd
import os
from django.conf import settings
from h5py._hl import dataset

path = os.path.join(settings.MEDIA_ROOT, 'crop_recommendation.csv')
dataset = pd.read_csv(path)
x = dataset.iloc[:, 3:7].values
y = dataset.iloc[:, 8].values

# Train and Test Data
from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(x, y,stratify=y, test_size=0.20, random_state=4)

# Scaling Features
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
x_test = sc_x.transform(x_test)


#Model neuralNetwork(using MLP Classifier)
def build_neural_model():
    from sklearn.neural_network import MLPClassifier
    nn=MLPClassifier(random_state=4,max_iter=550)
    nn.fit(x,y)
    y_pred=nn.predict(x_test)
    #performance Metrics
    from sklearn import metrics
    accuracy = metrics.accuracy_score(y, y_pred)
    precission = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1_score = metrics.f1_score(y, y_pred)
    print("NeuralNetwork accuracy : ", accuracy, precission, recall, f1_score)
    return accuracy, precission, recall, f1_score

