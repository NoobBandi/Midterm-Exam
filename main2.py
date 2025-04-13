import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

np.random.seed(47)

iris = datasets.load_iris()
print("資料集的特徵欄位名稱：", iris.feature_names)
print("資料集的目標值：", iris.target_names)
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
decison_tree_clf = DecisionTreeClassifier(criterion='entropy')
decison_tree_clf = decison_tree_clf.fit(X_train, Y_train)
Y_predict = decison_tree_clf.predict(X_test)
score = accuracy_score(Y_test, Y_predict)
print("鳶尾花分類的預測準確率：", score)

feature_names = ['花萼長', '花萼寬', '花瓣長', '花瓣寬']
dot_data = export_graphviz(decison_tree_clf, feature_names=feature_names, class_names=iris.target_names, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph