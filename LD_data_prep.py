#Import libraries
import pandas
import numpy as np
import statistics
import math
import os


from numpy import nan
from pandas import read_csv







# load and clean-up data

dataset = read_csv(r'household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

# summarize
print(dataset.shape)
print(dataset.head())

# mark all missing values
dataset.replace('?', nan, inplace=True)

# add a column for for the remainder of sub metering
values = dataset.values.astype('float32')
dataset['Sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])

# save updated dataset
dataset.to_csv('household_power_consumption.csv')

# load the new dataset and summarize
dataset = read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(dataset.head())



#Erase useless data

del dataset['Voltage']
del dataset['Global_intensity']
del dataset['Sub_metering_1']
del dataset['Sub_metering_2']
del dataset['Sub_metering_3']
del dataset['Sub_metering_4']

del dataset['Global_reactive_power']


#fill nan values with mean 
mean_value=dataset['Global_active_power'].mean()
dataset['Global_active_power'].fillna(value=mean_value, inplace=True)


#Calculate 15 minutes avg power 
#dataset.rename(columns = {'Global_reactive_power':'Active_power_15_min_avg'}, inplace = True)

minutes=15

dataset['Active_power_15_min_avg'] = pandas.Series([0 for x in range(len(dataset.index))])

#for i in range(dataset.Active_power_15_min_avg.size):
 #   dataset.Active_power_15_min_avg[i]=0

for i in range (6,dataset.Active_power_15_min_avg.size,minutes):
    dataset.Active_power_15_min_avg[i]=statistics.mean(dataset.Global_active_power[i:i+minutes-1])
    
#new dataset with only 15 minutes avarage 
df1=dataset.iloc[6::minutes]
del df1['Global_active_power']

df1 = df1.astype('float32') 
#df1 = df1.drop(index="2010-11-26 20:54:00")
df1.to_csv('Neural_Networks_data.csv') 


#df2=df1



del(dataset)
del(i)
del(minutes)
#del(df1)
del(mean_value)
del(values)