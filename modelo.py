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
#análisis exploratorio de datos
#gráfico de barras: supervivencia por sexo
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Supervivencia por Sexo')
plt.show()
#Histograma: Distribución de edad y supervivencia
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack')
plt.title('Distribución de edad y supervivencia')
plt.show()
#heatmap de la matriz de correlación
corr_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación')
plt.show()
#entrenamiento del modelo de regresión logística
#definir las características (x) y la variable objetivo (y)
X=df.drop('Survived', axis=1)
y=df['Survived']
#dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
#Inicializar y entrenar el modelo de regresión logística
model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
#hacer predicciones en el conjunto de prueba
predictions=model.predict(X_test)
#evaluación del modelo
