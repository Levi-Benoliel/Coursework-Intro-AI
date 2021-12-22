import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from fast_ml.model_development import train_valid_test_split




path = "../Coursework-Intro-AI/"



filename_read = os.path.join(path, "london_merged.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
df = df.select_dtypes(include=['int', 'float'])



#collect the columns names for non-target features

df = df [["cnt","t1","t2","hum","wind_speed","weather_code","is_holiday","is_weekend","season"]]

predict = "cnt"

X = np.array(df.drop([predict],1))
y = np.array(df[predict])


'''result = []
for x in df.columns:
    if x != 'cnt':
        result.append(x)

X = df[result].values
y = df['cnt'].values
'''

"""
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

"""




#split data into testing and training




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Xs_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)



"""X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.25, random_state=1)



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=1)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.25, random_state=1)
"""



print(X_train)
#print(Xs_train)




model = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()


model.fit(X_train, y_train)

acc = model.score(X_test, y_test)

print("accuracy: \n", acc)
print("Coefficient: \n", model.coef_)
print("Intercept: \n", model.intercept_)


predictions = model.predict(X_test)



for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])






plt.scatter(y_train, X_train[:,0])




























"""X_train, X_test, y_train, y_test, X_valid, y_valid, = train_valid_test_split(df, target = "cnt", test_size=0.1, random_state=0, valid_size = 0.1, train_size = 0.8)



X1_train, X1_test, y1_train, y1_test, X1_valid, y1_valid = train_valid_test_split(df, target = "cnt" test_size=0.1, random_state=0, valid_size = 0.1, train_size = 0.8)



X2_train, X2_test, y2_train, y2_test, X2_valid, y2_valid = train_valid_test_split(df, target = "cnt" test_size=0.1, random_state=0, valid_size = 0.1, train_size = 0.8)




X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0, stratify=df.output) # recommended configuration for training-test split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2



# build the model
model = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()


<<<<<<< HEAD
=======
#sample = np.array(y)
#sample2 = np.array(y1)
#sample3 = np.array(y2)
>>>>>>> dc3677f8ea0746d6083125cf6859e9e0c501057d

#model.predict(sample.reshape(-1,11))
#model2.predict(sample2.reshape(-1,11))
#model3.predict(sample3.reshape(-1,11))

<<<<<<< HEAD
#sample = np.array(y)
#sample2 = np.array(y1)
#sample3 = np.array(y2)



#model.predict(sample.reshape(-1,11))
#model2.predict(sample2.reshape(-1,11))
#model3.predict(sample3.reshape(-1,11))



=======
>>>>>>> dc3677f8ea0746d6083125cf6859e9e0c501057d
print(X_train.size)
print(X1_train.size)
print(X2_train.size)
print(y_train.size)
print(y1_train.size)
print(y2_train.size)



<<<<<<< HEAD


=======
>>>>>>> dc3677f8ea0746d6083125cf6859e9e0c501057d
print(model)
model.fit(X_train, y_train)
print(model.coef_)
print(model2)
model.fit(X1_train, y1_train)
print(model3)
model.fit(X2_train, y2_train)
print(model.coef_)


plt.scatter(X_train, y_train)



plt.scatter(X_train, y_train)"""