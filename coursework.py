import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
<<<<<<< HEAD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
=======

from sklearn import metrics
from fast_ml.model_development import train_valid_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
>>>>>>> 1dd41b3b953921e833b17ba3f6291f3166758ae0
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
path = "../Coursework-Intro-AI/"

filename_read = os.path.join(path, "london_merged.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
<<<<<<< HEAD

#fiter out timestamp and only incude integers and float values
df = df.select_dtypes(include=['int', 'float'])

#display number of null values
print(df.isnull().sum().sort_values(ascending=False))

#collect the columns names for non-target features

=======
df = df.select_dtypes(include=['int', 'float'])
>>>>>>> 1dd41b3b953921e833b17ba3f6291f3166758ae0
df = df [["cnt","t1","t2","hum","wind_speed","weather_code","is_holiday","is_weekend","season"]]
#count values stored as predict
predict = "cnt"

<<<<<<< HEAD


X = np.array(df.drop(["cnt","weather_code"],1))
y = np.array(df[predict])

#split data into testing and training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)




model = LinearRegression()
model2 = DecisionTreeRegressor(random_state = 10)
=======
X = np.array(df.drop([predict],1))
y = np.array(df[predict])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model2 = LinearRegression()

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis');

>>>>>>> 1dd41b3b953921e833b17ba3f6291f3166758ae0
model.fit(X_train, y_train)
model2.fit(X_train, y_train)

y_pred = model.predict(X_test)

<<<<<<< HEAD
acc = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
n = y_pred.shape[0]
k = X_train.shape[1]
adj_r_sq = 1 - (1 - r2)*(n-1)/(n-1-k)
print("mse: \n", acc/1000)
print("rmse: \n", rmse)
print("r2: \n", r2)
print("r2 adj: \n", adj_r_sq)

print("Coefficient: \n", model.coef_)
print("Intercept: \n", model.intercept_)
=======
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
>>>>>>> 1dd41b3b953921e833b17ba3f6291f3166758ae0

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#Predicted and actual values visualised in bar chart
predictions = model.predict(X_test)
y_pred = model.predict(X_test)
<<<<<<< HEAD
#create new dataframe df_compare
df_compare = pd.DataFrame({'Real': y_test, 'Predicted': y_pred})
#display 60 comparisons in visualisation
df_head = df_compare.head(60)
#define bar chart and dimensions
=======
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

>>>>>>> 1dd41b3b953921e833b17ba3f6291f3166758ae0
df_head.plot(kind='bar',figsize=(10,8))
#show plot
plt.show()

#definition for chart compapring size of actual values and size of predicted values
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='real')
    plt.plot(t['pred'].tolist(), label='predicted')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
<<<<<<< HEAD
#display comparison with 1000 results
chart_regression(y_pred[:1000].flatten(),y_test[:1000],sort=True)   
y_pred = model.predict(X_test)

#create another dataframe
df_compare = pd.DataFrame({'Real': y_test, 'Predicted': y_pred})
#display plot
sns.regplot(x='Real', y='Predicted', data=df_compare)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

n = y_pred.shape[0]
k = X_train.shape[1]
#formula for adjusted r2 squared
adj_r_sq = 1 - (1 - r2)*(n-1)/(n-1-k)

#display summary
print("mse: \n", mse/1000)
print("rmse: \n", rmse)
print("r2: \n", r2)
print("r2 adj: \n", adj_r_sq)


#alternate models


X = np.array(df.drop(["cnt", "wind_speed","weather_code","is_holiday","is_weekend",],1))
y = np.array(df[predict])

#split data into testing and training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#alternative model
modela = LinearRegression()
modela.fit(X_train, y_train)

#Predicted and actual values visualised in bar chart
predictionsa = modela.predict(X_test)
y_pred = modela.predict(X_test)
#create new dataframe df_compare
df_compare = pd.DataFrame({'Real': y_test, 'Predicted': y_pred})
#display 60 comparisons in visualisation
df_head = df_compare.head(60)
#define bar chart and dimensions
df_head.plot(kind='bar',figsize=(10,8))
#show plot
plt.show()
mse = mean_squared_error(y_test, y_pred)
print("mse: \n", mse/1000)

#neural network model training
modelS = Sequential()
modelS.add(Dense(20000, input_shape=X[1].shape, activation='relu')) 
modelS.add(Dense(1)) 
modelS.summary()
modelS.compile(loss='mean_squared_error', optimizer='adam')
modelS.fit(X_train,y_train,verbose=2,epochs=10)
modelS.summary()


#excluding humidity to see how it affects predictions
X = np.array(df.drop(["cnt","hum","wind_speed","weather_code","is_holiday","is_weekend",],1))
y = np.array(df[predict])

#split data into testing and training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#alternative model
modelb = LinearRegression()
modelb.fit(X_train, y_train)

#Predicted and actual values visualised in bar chart
predictionsb = modelb.predict(X_test)
y_pred = modelb.predict(X_test)
#create new dataframe df_compare
df_compare = pd.DataFrame({'Real': y_test, 'Predicted': y_pred})
#display 60 comparisons in visualisation
df_head = df_compare.head(60)
#define bar chart and dimensions
df_head.plot(kind='bar',figsize=(10,8))
#show plot
plt.show()
mse = mean_squared_error(y_test, y_pred)
print("mse: \n", mse/1000)
=======
chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)   

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.regplot(x='Actual', y='Predicted', data=df_compare)

print("accuracy: \n", acc)
>>>>>>> 1dd41b3b953921e833b17ba3f6291f3166758ae0
