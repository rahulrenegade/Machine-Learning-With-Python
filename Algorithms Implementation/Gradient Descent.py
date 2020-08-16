def gradientdecsent(X,y,theta,iterations):
	m=len(X)
	m_curr=b_curr=0
	iterations = 1500
	n = len(x)
	learningr = 0.01
	for i in range(iterations):
		y_predc=m_curr*x+b_curr
		jtheta=(1/n)*sum([val**2 for val in (y-y_predc)])
		md=-(2/n)*sum(x*(y-y_predc))
		bd=-(2/n)*sum(y-y_predc)
		m_curr=m_curr-learningr*md
		b_curr=b_curr-learningr*bd
		print(m_curr,b_curr,jtheta,i)
m_curr= np.zeros(2)
	