import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sonar_data = pd.read_csv(r"C:\Users\akhil gunti\OneDrive\Desktop\Machine Learning Projects\Machine-Learning-Projects\Sonar Rock VS Mine Prediction\Copy of sonar data.csv", header= None)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data.head(5)
sonar_data.describe()

sonar_data[60].value_counts()

x= sonar_data.drop(columns = 60, axis= 1)
y = sonar_data[60]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1, stratify= y,random_state=1)

model = LogisticRegression()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
bias = model.score(x_train,y_train)
variance = model.score(x_test,y_test)

#making a predictive system

input_data = (0.0228,	0.0853,	0.1,	0.0428,	0.1117,	0.1651,	0.1597,	0.2116,	0.3295,	0.3517,	0.333,	0.3643,	0.402,	0.4731,	0.5196,	0.6573,	0.8426,	0.8476,	0.8344,	0.8453,	0.7999,	0.8537,	0.9642,	1,	0.9357,	0.9409,	0.907,	0.7104,	0.632,	0.5667,	0.3501,	0.2447,	0.1698,	0.329,	0.3674,	0.2331,	0.2413,	0.2556,	0.1892,	0.194,	0.3074,	0.2785,	0.0308,	0.1238,	0.1854,	0.1753,	0.1079,	0.0728,	0.0242,	0.0191,	0.0159,	0.0172,	0.0191,	0.026,	0.014,	0.0125,	0.0116,	0.0093,	0.0012,	0.0036,
)
# changing the input_data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)
#reshape the np array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== "R"):
    print("Object is rock")
else:
    print("object is a mine")    
