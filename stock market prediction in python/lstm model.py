import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
start ='2010-01-01'
end='2019-12-31'
df=data.DataReader('AAPL','yahoo',start,end)
df=df.reset_index()
df=df.drop(['Date','Adj Close'],axis=1)
print(df.head())
print(plt.plot(df.Close))

ma100=df.Close.rolling(100).mean()
print(ma100)

plt.figure(figsize=(12,6))
plt.plot(df.Close)
print(plt.plot(ma100,'r'))


ma200=df.Close.rolling(200).mean()
print(ma200)


plt.figure(figsize=(12,6))
plt.plot(df.Close)
print(plt.plot(ma100,'r'))
print(plt.plot(ma200,'g'))
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)
print(data_training_array) 

x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100 :i])
    y_train.append(data_training_array[i,0])

print(x_train)

x_train,y_train=np.array(x_train),np.array(y_train)
model =Sequential()
model.add(LSTM(units=50,activation="relu",return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))


model.add(LSTM(units=60,activation="relu",return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation="relu",return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# model.summary()