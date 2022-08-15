"""
Created on Fry Jul 08  12:01:34 2022

@author: MB6372
"""

import keras
import sklearn
import matplotlib.pyplot as plt
import pandas
import numpy as np
import math
import os
import tensorflow.python


from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.layers import ConvLSTM2D
from pandas import read_csv

print(tensorflow.__version__)


#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer


df1=read_csv('Neural_Networks_data.csv')

df = df1[["Active_power_15_min_avg"]].to_numpy()
df = scaler.fit_transform(df)

#To reduce training time for preliminary tests
#df=df[1:1001]

train_size = int(len(df) * 0.80)
test_size = len(df) - train_size
train, test = df[0:train_size,:], df[train_size:len(df),:]

def to_sequences(df, seq_size=1):
    x = []
    y = []

    for i in range(len(df)-seq_size-1):
        #print(i)
        window = df[i:(i+seq_size), 0]
        x.append(window)
        y.append(df[i+seq_size, 0])
        
    return np.array(x),np.array(y)


#pred_size=2
seq_size =2  # Number of time steps to look back 96=1 day , 192=2 days
#Larger sequences (look further back) may improve forecasting.

trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)

print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))



#print(trainX.shape[0])



###################################################
#ConvLSTM
#The layer expects input as a sequence of two-dimensional images, 
#therefore the shape of input data must be: [samples, timesteps, rows, columns, features]

trainX = trainX.reshape((trainX.shape[0], 1, 1, 1, seq_size))
testX = testX.reshape((testX.shape[0], 1, 1, 1, seq_size))



model_LSTM = Sequential()
model_LSTM.add(ConvLSTM2D(filters=64, kernel_size=(1,1), activation='relu', input_shape=(1, 1, 1, seq_size)))
model_LSTM.add(Flatten())
model_LSTM.add(Dense(32))
model_LSTM.add(Dense(1))
model_LSTM.compile(optimizer='adam', loss='mean_squared_error')
model_LSTM.summary()
######################################################

'''

######################################################
#Bidirectional LSTM
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#
##For some sequence forecasting problems we may need LSTM to learn
## sequence in both forward and backward directions

model_LSTM = Sequential()
model_LSTM.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(None, seq_size)))
model_LSTM.add(Dense(1))
model_LSTM.compile(optimizer='adam', loss='mean_squared_error')
model_LSTM.summary()

#print('Train...')
######################################################
'''


model_LSTM.fit(trainX, trainY, validation_data=(testX, testY),
          verbose=2, epochs=10)


# make predictions

trainPredict = model_LSTM.predict(trainX)
testPredict = model_LSTM.predict(testX)




# invert predictions back to prescaled values

#This is to compare with original input values
#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])



for i in range(trainPredict.size):
        if trainPredict[i]<0:
            trainPredict[i]=0
            
for i in range(testPredict.size):
        if testPredict[i]<0:
            testPredict[i]=0

#Predict=trainPredict.append(testPredict)

testY=np.roll(testY,1)
trainY=np.roll(trainY,1)
#testY[0]=testY[1]
#trainY[0]=trainY[1]
#print(testY[1])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(df)-1, :] = testPredict


# plot baseline and predictions
plt.plot(scaler.inverse_transform(df),label='LSTM')
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
#plt.leggend()
plt.show()

'''
plt.plot(testY.transpose()[1000:1200])
plt.plot(testPredict[1000:1200])
plt.show()



day_predict=np.zeros_like(testX[0])
day_predict=day_predict.reshape(pred_size)

index=1260
day_inputs=testX[index]

day_inputs=day_inputs.reshape((1, 1, 1, 1, seq_size))


for i in range(pred_size):
    day_predict[i]=model_LSTM.predict(day_inputs)
    day_inputs=np.roll(day_inputs,-1)
    day_inputs[0,0,0,-1]=day_predict[i]
    


day_predict=day_predict.reshape(pred_size,1)
day_predict=scaler.inverse_transform(day_predict)
day_predict=day_predict.reshape(pred_size)

plt.plot(testY.transpose()[index:index+pred_size])
plt.plot(day_predict)
plt.show()
'''