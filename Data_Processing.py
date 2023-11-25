import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def produce_data(Test,Train):
    Test_data = pd.read_table("DATA/"+Test, sep=',')
    Train_data = pd.read_table("DATA/"+Train, sep=',')
    user_col = [i for i in range(Test_data.shape[1])]
    Test_data = pd.read_table("DATA/"+Test, sep=',', header=None, names=user_col)
    Train_data = pd.read_table("DATA/" + Train, sep=',', header=None, names=user_col)
    Test_data[Test_data.shape[1]-1] = pd.Categorical(Test_data[Test_data.shape[1]-1]).codes
    Train_data[Train_data.shape[1] - 1] = pd.Categorical(Train_data[Train_data.shape[1] - 1]).codes
    return Test_data,Train_data
def data_to_numpy(Test,Train):
    Test,Train = produce_data(Test,Train)
    Test_label = Test[Test.shape[1]-1].to_numpy().reshape(Test.shape[0],1)
    Train_label = Train[Train.shape[1] - 1].to_numpy().reshape(Train.shape[0], 1)
    X_train = Train.to_numpy()
    Y_test = Test.to_numpy()
    return Y_test,Test_label ,X_train,Train_label

