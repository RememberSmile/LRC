import numpy as np
def Get_class_data(class_index, x_train,class_data):
    for i in range(len(class_index)):
         for j in range(len(class_index[i])):
             class_data.append(x_train[class_index[i][j], :])
