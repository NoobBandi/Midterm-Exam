import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(47)

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
random_forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
random_forest_clf = random_forest_clf.fit(X_train, Y_train)
Y_predict = random_forest_clf.predict(X_test)
score = accuracy_score(Y_test, Y_predict)
print("鳶尾花分類的預測準確率：", score)