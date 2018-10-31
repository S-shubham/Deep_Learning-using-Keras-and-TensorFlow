#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df =pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
test_data_df =pd.read_csv("sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler =MinMaxScaler(feature_range=(0,1))  

# Scale both the training inputs and outputs
scaled_training =scaler.fit_transform(training_data_df)
scaled_testing =scaler.transform(test_data_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("sales_data_test_scaled.csv", index=False)


# In[14]:


import sys


# In[24]:


from keras.models import Sequential
from keras.layers import *

training_data_df=pd.read_csv("sales_data_training_scaled.csv")

X=training_data_df.drop('total_earnings',axis=1).values
Y=training_data_df[['total_earnings']].values


model=Sequential()
model.add(Dense(50,input_dim=9, activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X,Y, epochs=50,shuffle=True,verbose=2)

# load the testing data sets

test_data_df=pd.read_csv("sales_data_test_scaled.csv")

X_test=test_data_df.drop('total_earnings',axis=1).values
Y_test=test_data_df[['total_earnings']].values

test_error_rate=model.evaluate(X_test,Y_test,verbose=0)

print("MSE for test data sets is: {}".format(test_error_rate))


# In[28]:


# predicting the value using our trained model

xx=pd.read_csv("proposed_new_product.csv").values

prediction=model.predict(xx)

prediction=prediction[0][0]

prediction+=0.115913
prediction/=0.0000036968

print("Earning prediction is: ${}".format(prediction))


# In[29]:



# saving the trained model to a file
model.save("trained_model.h5")


# In[33]:



# we can now load this trained model for reuse
# the trained model will store the structure of the neurall network as well as the weight values for each neuron
from keras.models import load_model
New_model=load_model("trained_model.h5")


# In[ ]:




