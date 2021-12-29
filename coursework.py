import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
path = "../Coursework-Intro-AI/"



filename_read = os.path.join(path, "london_merged.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

#fiter out timestamp and only incude integers and float values
df = df.select_dtypes(include=['int', 'float'])

#display number of null values
print(df.isnull().sum().sort_values(ascending=False))

#collect the columns names for non-target features

df = df [["cnt","t1","t2","hum","wind_speed","weather_code","is_holiday","is_weekend","season"]]
#count values stored as predict
predict = "cnt"



X = np.array(df.drop(["cnt","weather_code"],1))
y = np.array(df[predict])

#split data into testing and training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)




model = LinearRegression()
model2 = DecisionTreeRegressor(random_state = 10)
model.fit(X_train, y_train)
model2.fit(X_train, y_train)

y_pred = model.predict(X_test)

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


#Predicted and actual values visualised in bar chart
predictions = model.predict(X_test)
y_pred = model.predict(X_test)
#create new dataframe df_compare
df_compare = pd.DataFrame({'Real': y_test, 'Predicted': y_pred})
#display 60 comparisons in visualisation
df_head = df_compare.head(60)
#define bar chart and dimensions
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