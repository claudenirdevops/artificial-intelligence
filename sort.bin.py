# Passo 1: Use o dataset breast_cancer da biblioteca Scikit-Learn para classificar tumores como "malignos" ou "benignos"

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


# Carregar o dataset

cancer = datasets.load_breast_cancer()

# Passo 2: Divida o conjunto de dados em 80% para treinamento e 20% para teste

X_train, X_test, y_train, y_test =

train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

Passo 3: Treine um modelo SVM para realizar a classificação binária

model = SVC(kernel='linear')

model.fit(X_train, y_train)

O resultado do treinamento do código do Passo 3, gera a classificação como na Figura 4.




# Passo 4: Calcule a acurácia do modelo no conjunto de teste

accuracy = model.score(X_test, y_test)

print(f'Acurácia do Modelo: {accuracy:.2f}')