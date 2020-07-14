import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
def warmupExercise():
	A=np.eye(4)
	print(A)
	return A
warmupExercise()
data = np.loadtxt(os.path.join('C:/Users/mulla/Desktop/values.txt'),delimiter=',')
X, y=data[:,0],data[:,1]
m=y.size
def plotdata(x,y):
	plt.plot(x,y,'ro',ms=10,mec='k')
	plt.xlabel('the profit 10,000$s')
	plt.ylabel('population in 10,000s')
	plt.show()
plotdata(X,y)
X = np.stack([np.ones(m), X], axis=1)
def computecost(x,y,theta):
	n= len(x)
	y_predc=np.dot(X, theta)
	temp = np.power((y-y_predc), 2)
	jtheta=np.sum(temp) / (2 * n)
	return jtheta
jtheta =computecost(X,y,theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % jtheta)
print('Expected cost value (approximately) 32.07\n')
jtheta = computecost(X, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2fr' % jtheta)
print('Expected cost value (approximately) 54.24')
def gradientdecsent(x,y,theta,alpha):
	iterations=1500
	m = len(X)
	theta = theta.copy()
	J_history = [] 
	for i in range(iterations):
		theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
		J_history.append(computecost(X, y, theta))
	return theta, J_history
theta=np.zeros(2)
alpha=0.01
theta,J_history=gradientdecsent(X,y,theta,alpha)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')
plotdata(X[:,1],y)
plt.plot(X[:,1],np.dot(X, theta),'-')
plt.legend(['Training data','Lining Regression'])
fprediction=np.dot([1,3.5],theta)
print('the predicted profit for 35,000 people is {:.2f}\n'.format(fprediction*10000))
sprediction=np.dot([1,7],theta)
print('the predicted profit for 70000 people is {:.2f}\n'.format(sprediction*10000))
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computecost(X, y, [theta0, theta1])
J_vals = J_vals.T
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')
ax = plt.subplot(122)
plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, showing minimum')
plt.show()
data = np.loadtxt(os.path.join('C:/Users/mulla/Desktop/values2.txt'),delimiter=',')
X, y=data[:,:2],data[:,2]
m = y.size
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
def featurenormalization(X):
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    result = (X - mean) / stddev
    return result, mean , stddev
result, mean, stddev = featurenormalization(X)
print('Computed mean of features:',mean)
print('computed standard deviation of features:',stddev)
X= np.concatenate([np.ones((m, 1)),result], axis=1)
def computecostmulti(X, y, theta):
    m=len(X)
    jtheta=0
    y_predc=np.dot(X,theta)
    jtheta=(1/2*m)*np.sum(np.square(np.dot(X,theta)-y))
    return jtheta
def gradientdescentmulti(X, y,theta,alpha):
	iterations=1500
	m = len(X)
	theta = theta.copy()
	J_history = [] 
	for i in range(iterations):
		theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
		J_history.append(computecost(X, y, theta))
	return theta, J_history
data = np.loadtxt(os.path.join('C:/Users/mulla/Desktop/values2.txt'),delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)
def normaleq(X,y):
	m=y.size
	theta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
	return theta
theta=normaleq(X,y)
X_array=[1,1650,3]
X_array[1:3] = (X_array[1:3] - mean) / stddev
price=np.dot(X_array,theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))
print(X_array[1:3])
print(theta)

