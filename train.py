#import essential libraries
import pandas as pd
import numpy as np
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt

 

#read dataset using pandas
data = pd.read_csv("Merged_stress_data.csv")
#printing rows
print(data.head())
print(data.shape)
#getting columns
print(data.columns)

#data division
y_final = data['Stress']
x_final = data.drop(['Stress'], axis=1)

print(x_final)
print(y_final)



# TRAIN - TEST SPLITTING
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2)
print("\nTraining set")
print(x_train.shape)
print(y_train.shape)
print("\nTesting set")
print(x_test.shape)
print(y_test.shape)



#standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
pickle.dump(scaler,open('Models/scaler.pkl','wb'))

from sklearn.ensemble import RandomForestClassifier

#initialize KNN classifier
classifier2=RandomForestClassifier()
#training
classifier2.fit(x_train, y_train)
#prediction on test dataset
y_pred2=classifier2.predict(x_test)

conf_matrix2 = confusion_matrix(y_true=y_test, y_pred=y_pred2)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix2, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig('Extra/confusion_matrix.png')
plt.show()

# calculate accuracy
acc2 = accuracy_score(y_test, y_pred2)
print(f"RF Accuracy :{round(acc2,3)*100}%")

# Calculate the precision
precision = precision_score(y_test, y_pred2, average="micro")
print(f"RF Precision :{round(precision,3)*100}%")

filename2 = "Models/RF_model.sav"
joblib.dump(classifier2, filename2)

