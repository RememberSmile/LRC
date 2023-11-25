from sklearn.linear_model import Lasso,ridge_regression
def Calculate_L1(X,Y,a):
    model = Lasso(alpha=a)
    model.fit(X, Y)
    w = model.coef_
    return w

def Calculate_L2(X,Y,c):
    model = ridge_regression(X,Y,alpha=c)
    return model