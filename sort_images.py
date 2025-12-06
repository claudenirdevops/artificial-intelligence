# Passo 1: Importe o dataset digits da biblioteca Scikit-Learn

from sklearn import datasets
digits = datasets.load_digits()

# Passo 2: Visualize os dados com matplotlib³

import matplotlib.pyplot as plt

plt.imshow(digits.images[0], cmap='gray')

plt.title(f'Label: {digits.target[0]}')

plt.show()

#Passo 3: Treine um modelo de classificação utilizando as Máquinas de Vetores de Suporte - Support Vector Machines (SVM)

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


X_train, X_test, y_train, y_test =
train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

model = SVC(gamma=0.001)

model.fit(X_train, y_train)

# Passo 4: Avalie o desempenho do modelo

accuracy = model.score(X_test, y_test)

print(f'Acurácia do Modelo: {accuracy:.2f}')