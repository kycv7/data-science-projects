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
#llenar los valores nulos en 'Age' con la media de la edad
df['Age'].fillna(df['Age'].mean(), inplace=True)
#llenar los valores nulos en 'Embarked' con el valor más común (moda)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
#convertir las variables categóricas 'sex' y 'embarked' a numéricas
df['Sex']=df['Sex'].map({'male':0,'female':1})
df=pd.get_dummies(df,columns=['Embarked'], drop_first=True)
#eliminar columnas no necesarios para el modelo
df=df.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
print(df.isnull().sum()) #verificar que no hay más nulos
print(df.head())
