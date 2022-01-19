''' Import Libraries '''
import os
from tensorflow.keras.models import load_model
import numpy as np


curr_path = os.getcwd()
classes = ['Conduction Disturbance', 'Hypertrophy', 'Myocardial Infarcation', 'Normal ECG', 'ST/T change']


''' Load a sample file '''
x = np.loadtxt(curr_path + "/sample files/sttc.csv", delimiter=",")
x = x.transpose(1, 0)                              # transpose matrix
x = np.expand_dims(x, axis=(0, -1))                # Add another channel on left and right


''' Load Model '''
model = load_model(curr_path + '/trained models/ST-CNN-GAP-5.h5')


''' Evaluate Model '''
y_prob = model.predict(x)

print("Sample belongs to following classes:")
for i in range(5):
    if y_prob[0][i] >= 0.5 :
        print(" * " + classes[i])
