import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(29)

california = fetch_california_housing(as_frame=True)
print("資料集的特徵欄位名稱：", california.feature_names)
X = california.data.values
Y = california.target.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
regression_clf = LinearRegression()
regression_clf.fit(X_train, Y_train)
Y_predict = regression_clf.predict(X_test)
score = r2_score(Y_test, Y_predict)
print("房價的預測準確表：", score)

plt.plot(Y_test[:200], label='實際房價')
plt.plot(Y_predict[:200], label='預測房價')
plt.legend()