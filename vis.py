import matplotlib.pyplot as plt
import numpy as np

MSG_SIZE = 2

op_data = np.loadtxt(open('.csv', 'rb'), delimiter=',', skiprows=1)

# sort by message size
op_data[op_data[:, MSG_SIZE].argsort()]

# https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array
data_by_msg_size = np.split(op_data, np.where(np.diff(op_data[:, MSG_SIZE]))[0] + 1)

for i in data_by_msg_size:
    
