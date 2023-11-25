import numpy as np
def grad_f(X,Y, W, rho2, L):
    xtx = np.matmul(np.transpose(X), X)
    a = np.matmul(xtx, W) + np.matmul(np.transpose(xtx), W)
    b = a - np.transpose(np.matmul(np.transpose(Y),X))-np.matmul(np.transpose(X),Y)
    d = np.matmul(np.matmul(np.transpose(X), L), X)
    e = rho2 * np.matmul((d + np.transpose(d)), W)
    return e + b
def prox_h(W,t,mu):
    return np.sign(W) * np.maximum(abs(W) - t * mu, 0)
def APG(X,Y,x_init,rho1,rho2,L,max_iters=2500,eps=1e-3,alpha=1.01,beta=0.5,use_restart=True):
    x = np.copy(x_init)
    y = np.copy(x_init)
    g = grad_f(X,Y, y, rho2, L)           
    theta = 1.
    t = 1. / np.linalg.norm(g,ord='fro') 
    x_hat = x - t * g
    g_hat = grad_f(X,Y, x_hat, rho2, L)
    t = abs(np.vdot((x - x_hat),(g - g_hat)) / np.linalg.norm(g - g_hat,ord='fro') ** 2)

    errs = np.zeros(max_iters)          
    k = 0
    err1 = np.nan

    for k in range(max_iters):

        x_old = np.copy(x)
        y_old = np.copy(y)

        x = y - t*g                     

        x = prox_h(x,t,rho1)            

        err1 = np.linalg.norm(y-x,ord='fro')/(1+np.linalg.norm(x,ord='fro'))/t

        y = x + (1-theta)*(x-x_old)

        g_old = np.copy(g)
        g = grad_f(X,Y, y, rho2, L)


        errs[k]=err1                  

        if err1<eps:         
            break
        theta = 2./(1+np.sqrt(1+4/(theta**2)))
        if use_restart and np.vdot((y-x),(x-x_old))>0:
            x = np.copy(x_old)
            y = np.copy(x)
            theta = 1.
        else:
            t_old = t
            t_hat = 0.5*(np.linalg.norm(y - y_old,ord='fro')**2)/abs(np.vdot((y-y_old),(g_old-g)))
            t = min(alpha*t,max(beta*t,t_hat))

    return x





