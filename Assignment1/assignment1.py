def featureNormalization(X):
    """ Feature Normalization """
    m = np.mean(X)
    s = np.std(X)
    normilezed_X = (X-m)/s
    return normilezed_X

def normalEquation(X, y):
    # Normal Equations
    XT = np.transpose(X)
    XTX = XT.dot(X)
    inv = np.linalg.pinv(XTX)
    temp = inv.dot(XT)
    theta = temp.dot(y)
    return theta

def computeCost(X, y, theta=[[0], [0]]):
    """ Computing Cost (for Multiple Variables) """
    J = 0
    m = y.size
    h = X.dot(theta)
    J = (1/(2*m)) * np.sum(np.square(h-y))
    return(J)

#might edit
def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    """ Gradient Descent (for Multiple Variables) """
    # J_history 
	m = y.size
    J_history = []
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1/m)*(X.T.dot(h-y))
        J_history.append(computeCost(X, y, theta))
    return(theta, J_history)

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def costFunction(theta, X, y):
    """ Logistic Regression Cost """
    XO = X.dot(theta)
    A = np.log(sigmoid(XO))
    ATY = A.transpose().dot(y)
    B = np.log(1-sigmoid(XO))
    BTY = B.transpose().dot(1-y)
    J = (ATY + BTY)/X.shape[0]
    return J

def gradient(theta, X, y):
    """ Logistic Regression Gradient """
    theta = theta.reshape(3,1)
    XT = X.transpose()
    XO = X.dot(theta)
    G = sigmoid(XO)
    Y = (-1)*y
    GYsum = G + Y
    grad = (XT.dot(GYsum))/X.shape[0]
    return grad.flatten()

def predict(theta, X, threshold=0.5):
    """ Logistic Regression predict """
    XO = X.dot(theta)
    ls = sigmoid(XO)>=threshold
    return ls.astype(int)

def costFunctionReg(theta, reg, *args):
    """ Regularized Logistic Regression Cost """
    X = args[0]
    y = args[1]
    XO = X.dot(theta)
    ATY = np.log(sigmoid(XO)).transpose().dot(y)
    BTY = np.log(1-sigmoid(XO)).transpose().dot(1-y)
    sumLeft = -(ATY + BTY)/X.shape[0]
    theta[0] = 0
    lsTheta = theta*theta
    sumRight = np.sum(lsTheta) * (reg/(2*X.shape[0]))
    costFn = (sumLeft+sumRight)
    return costFn[0]

def gradientReg(theta, reg, *args):
    """ Regularized Logistic Regression Gradient """
    X = args[0]
    y = args[1]
    theta = theta.reshape(theta.shape[0],1)
    XT = X.transpose()
    G = sigmoid(X.dot(theta))
    Y = (-1)*y
    GYsum = G + Y
    sumLeft = (XT.dot(GYsum))/X.shape[0]
    theta[0] = 0
    sumRight = (theta)*(reg/X.shape[0])
    grad = sumLeft+sumRight
    grad = grad.flatten()
    return grad
