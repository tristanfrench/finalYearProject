import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.signal
#data1 = open('train_500_theta2.pickle', 'rb')

data1 = open('train_500_theta_error.pickle', 'rb')
a = pickle.load(data1)
a=a[1:]
a[25]=(a[24]+a[26])/2
data2 = open('val_500_theta2_error.pickle', 'rb')
b = pickle.load(data2)
b=b[1:]
'''
sg_n = 21
#a=scipy.signal.savgol_filter(a, sg_n, 3)
b=scipy.signal.savgol_filter(b, sg_n, 3)
plt.plot(a,label='Train')
plt.xlabel('Mean Squared Error Loss')
plt.ylabel('Number of Epochs')

plt.plot(b,label='Validation')

plt.legend(loc='upper left')
plt.show()
'''
#plt.plot(np.gradient(a[35:]))
n=50
a_interval = []
for idx,i in enumerate(a):
    temp_mean = np.mean(a[idx:idx+n])
    if len(a[idx:idx+n]) == n:
        a_interval.append(temp_mean)
    else:
        break
plt.plot(a_interval)
plt.xlabel('Starting Index of the Interval')
plt.ylabel('Mean Validation Loss over an Interval')
plt.show()

#data2 = open('val_500_theta2.pickle', 'rb')



b_interval = []
for idx,i in enumerate(b):
    temp_mean = np.mean(b[idx:idx+n])
    if len(b[idx:idx+n]) == n:
        b_interval.append(temp_mean)
    else:
        break
plt.plot(b_interval)
plt.xlabel('Starting Index of the Interval')
plt.ylabel('Mean Validation Loss over an Interval')
plt.show()


