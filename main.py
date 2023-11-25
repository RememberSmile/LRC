import numpy as np
import Data_Processing as DP
from Data_norm import normalise_data
from Calculate_info import Calculate_Index
from Calculate_Regression import Calculate_L1,Calculate_L2
from Get_class_data import Get_class_data
from LPP import output_L
from APG import APG
from sklearn.metrics import accuracy_score
from tqdm import tqdm



rho1 = 0.0001
rho2 = 0.01
rho3_lrc1 = 0.001
rho3_lrc2 = 1
TEST_NAME = "Demo-test.csv"
TRAIN_NAME = "Demo-train.csv"


Y_test, Y_label, X_train, X_label = DP.data_to_numpy(TEST_NAME,TRAIN_NAME)
class_number = [t for item in Y_label for t in item]
class_number = list(set(class_number))

print("DATA:",TRAIN_NAME)

class_index = []
class_data = []

X_train = np.delete(X_train, -1, axis=1)
X_train = normalise_data(X_train)
X_train = np.c_[X_train, X_label]

for i in range(len(class_number)):
    samples_index = list(np.where(X_train[:,-1] == class_number[i])[-1:])
    class_index.append(samples_index)
Get_class_data(class_index, X_train, class_data)


Index = []
for i in tqdm(range(len(class_index))):
    class_data[i] = np.delete(class_data[i], -1, axis=1)
    class_data[i] = np.transpose(class_data[i])
    L_t = output_L(class_data[i], 5)
    x_init = np.random.randn(class_data[i].shape[1],class_data[i].shape[1])
    Weight = APG(class_data[i],class_data[i],x_init,rho1,rho2,L_t)
    Index.append(Calculate_Index(Weight))
print("Objective function optimization completed")

Y_test = np.delete(Y_test, -1, axis=1)
Y_test = np.transpose(normalise_data(Y_test))
X_train = np.delete(X_train, -1, axis=1)
X_train = np.transpose((X_train))

#LRC_1
result_pre = []
for i in tqdm(range(Y_test.shape[1])):
    result = []
    residual = []
    index_of_residual = []

    w = Calculate_L1(X_train, Y_test[:, i],rho3_lrc1)
    w_re = w.reshape(w.shape[0], 1)
    w_list = [t for item in w_re for t in item]

    for q in range(len(class_number)):
        temp = []
        for j in range(len(class_index[q][0])):
            temp.append(w_list[class_index[q][0][j]])
        index_of_max_w = temp.index(max(temp))

        Close_point = index_of_max_w
        Index_of_Close_point = Index[q][Close_point]
        index_of_residual.append(Index_of_Close_point)

        for t in range(len(Index_of_Close_point)):
            residual.append(class_data[q][:, Index_of_Close_point[t]])

    w_last = Calculate_L1(np.transpose(np.array(residual)), Y_test[:, i],rho3_lrc1)
    w_last_re = w_last.reshape(w_last.shape[0], 1)
    w_last_list = [t for item in w_last_re for t in item]

    residual_result = []
    for r in range(len(class_number)):
        local = []
        local_vector = []
        for u in range(len(index_of_residual[r])):
            local.append(w_last_list[u])
            local_vector.append(np.transpose(np.array(residual))[:, u])
        y = np.matmul(np.transpose(local_vector), np.array(local))
        value_of_residual = np.sqrt(np.sum(np.square(y-Y_test[:,i])))
        sum_weight = 1 / value_of_residual
        result.append(sum_weight)
        residual_result.append(value_of_residual)
        w_last_list = w_last_list[len(index_of_residual[r]):]
        residual = residual[len(index_of_residual[r]):]

    result_pre.append(class_number[result.index(max(result))])
print("LRC1 acc:",accuracy_score(Y_label,result_pre))

#LRC_2
result_pre = []

for i in tqdm(range(Y_test.shape[1])):
    result = []
    residual = []
    index_of_residual = []

    w = Calculate_L2(X_train, Y_test[:, i],rho3_lrc2)
    w_re = w.reshape(w.shape[0], 1)
    w_list = [t for item in w_re for t in item]

    for q in range(len(class_number)):
        temp = []
        for j in range(len(class_index[q][0])):
            temp.append(w_list[class_index[q][0][j]])
        index_of_max_w = temp.index(max(temp))

        Close_point = index_of_max_w
        Index_of_Close_point = Index[q][Close_point]
        index_of_residual.append(Index_of_Close_point)

        for t in range(len(Index_of_Close_point)):
            residual.append(class_data[q][:, Index_of_Close_point[t]])

    w_last = Calculate_L2(np.transpose(np.array(residual)), Y_test[:, i],rho3_lrc2)
    w_last_re = w_last.reshape(w_last.shape[0], 1)
    w_last_list = [t for item in w_last_re for t in item]

    residual_result = []
    for r in range(len(class_number)):
        local = []
        local_vector = []
        for u in range(len(index_of_residual[r])):
            local.append(w_last_list[u])
            local_vector.append(np.transpose(np.array(residual))[:, u])
        y = np.matmul(np.transpose(local_vector), np.array(local))
        value_of_residual = np.sqrt(np.sum(np.square(y-Y_test[:,i])))
        sum_weight = np.sqrt(np.sum(np.array(local) ** 2)) /  value_of_residual
        result.append(sum_weight)
        residual_result.append(1/sum_weight)
        w_last_list = w_last_list[len(index_of_residual[r]):]
        residual = residual[len(index_of_residual[r]):]
    result_pre.append(class_number[result.index(max(result))])

print("LRC2 acc",accuracy_score(Y_label,result_pre))