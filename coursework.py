import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
path = "../Coursework-Intro-AI/"



filename_read = os.path.join(path, "london_merged.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])
df = df.select_dtypes(include=['int', 'float'])



#collect the columns names for non-target features

df = df [["cnt","t1","t2","hum","wind_speed","weather_code","is_holiday","is_weekend","season"]]

predict = "cnt"

#np.array(df.drop(["hum","wind_speed","weather_code","is_holiday","is_weekend"],1))






#X = df.iloc[:,[2]].values
#y = df.iloc[:, :1].values
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

"""
std = StandardScaler()
afp = np.append(X_train[].values, X_test['AFP'].values)
std.fit(afp)

X_train[['cnt']] = std.transform(X_train['cnt'])
X_test[['cnt']] = std.transform(X_test['cnt'])
"""



"""X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.25, random_state=1)



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=1)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.25, random_state=1)
"""

'''
model2 = LinearRegression()
thise = model2.fit(X_train, y_train)
'''
#print(X_train)
#print(Xs_train)

'''
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

plt.scatter(X_train[:,0], X_train[:,0], color = 'red', alpha = 0.2)
#plt.plot(X_train, thise.predict(X_train), color = 'green')
plt.show()
'''


model = LinearRegression()
model2 = LinearRegression()
#model3 = Sequential()


from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis');

model.fit(X_train, y_train)
acc = model.score(X_test, y_test)


from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


#print("Coefficient: \n", model.coef_)
#print("Intercept: \n", model.intercept_)


predictions = model.predict(X_test)
y_pred = model.predict(X_test)
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)



df_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#for x in range(len(predictions)):
   # print(predictions[x], X_test[x], y_test[x])

def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)   

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.regplot(x='Actual', y='Predicted', data=df_compare)



'''
plt.scatter(X_test, X_test)
#plt.plot(X_train, model.predict(X_train), color = "green")
plt.title('count (Training set)')
plt.xlabel('')
plt.ylabel('count')
plt.show()
'''
'''
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(X_train,y_train,verbose=2,epochs=100)
model2.summary()
model2.add(Dense(1024, input_shape=X[1].shape, activation='sigmoid')) # Hidden 1
model2.add(Dense(1)) # Output
model2.summary()
#a = df.iloc[:, :4].values
'''
#print (a)

print("accuracy: \n", acc)
'''
print(df.isnull().sum().sort_values(ascending=False))
print(X)

'''











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