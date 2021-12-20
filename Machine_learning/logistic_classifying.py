from re import search
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Neuron_for_logistic_classifying import Neuron

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.2, random_state=42)

neuron = Neuron()

neuron.fit(x_train, y_train, -1)
portion = np.mean(neuron.predict(x_test) == y_test)
print(portion)