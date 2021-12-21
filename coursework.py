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

#print to check that this has worked
print(df[:5]) 

#collect the columns names for non-target features
result = ["weather_code"]
for x in df.columns:
    if x != 'cnt':
        result.append(x)
   
X = df[result].values
y = df['cnt'].values


#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



# build the model
model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)

