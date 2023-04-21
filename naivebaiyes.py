import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("resto_preprocessed.csv", delimiter=";")
x = dataset[["subdistrict", "Average Price"]]
y = dataset["Category_Bakmie"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
correct_predictions = np.trace(cm)
print(correct_predictions)
total_samples = len(y_test)
print(total_samples)
accuracy = correct_predictions / total_samples
print("Accuracy : ", accuracy)

X_set, y_set = np.concatenate(
    (x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("red", "green"))(i), label=j)
plt.title("Gaussian Naive Bayes (Test set)")
plt.xlabel("Subdistrict")
plt.ylabel("Average Price")
plt.legend()
plt.show()
