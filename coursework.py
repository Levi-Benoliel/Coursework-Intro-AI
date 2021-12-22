import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
path = "../Coursework-Intro-AI/"

filename_read = os.path.join(path, "london_merged.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
df = df.select_dtypes(include=['int', 'float'])

#collect the columns names for non-target features
result = ["season", "t1", "t2"]
for x in df.columns:
    if x != 'cnt':
        result.append(x)
   
X = df[result].values
y = df['cnt'].values

result = ["hum", "wind_speed", "weather_code"]
for x in df.columns:
    if x != 'cnt':
        result.append(x)
   
X1 = df[result].values
y1 = df['cnt'].values

result = ["t1", "t2", "hum"]
for x in df.columns:
    if x != 'cnt':
        result.append(x)
   
X2 = df[result].values
y2 = df['cnt'].values


#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.5, random_state=0)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.5, random_state=0)


# build the model
model = LinearRegression()
print("Model1: ")
model.fit(X_train, y_train)
print(model.coef_)
print('\n')
print("Model2: ")
model.fit(X1_train, y1_train)
print(model.coef_)
print('\n')
print("Model3: ")
model.fit(X2_train, y2_train)
print(model.coef_)




