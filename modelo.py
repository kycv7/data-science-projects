#cargar datos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import kagglehub
path=kagglehub.dataset_download("yasserh/titanic-dataset")
print("path to dataset files: ", path)
import os
path = r"C:\Users\cklyf\.cache\kagglehub\datasets\yasserh\titanic-dataset\versions\1"
print(os.listdir(path))
df=pd.read_csv(path + "/Titanic-Dataset.csv")
print(df.head())
#limpiar y transformar datos
#Eliminar la columna 'Cabin' debido a la gran cantidad de valores nulos
df=df.drop('Cabin', axis=1)
