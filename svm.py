import math
import random
import csv

def get_data(name):
	data = []
	with open(name, 'r') as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			for i in range(len(row)-1):
				row[i] = float(row[i])
			row[-1] = int(row[-1])
			data = data + [row]
	return data

def get_labels(data):
	labels = []
	for i in range(len(data)):
		labels = labels + [data[i][-1]]
		data[i] = data[i][:-1]
	return labels

def kernel_linear(x_1,x_2):
	ans = float(0)
	for i in range(len(x_1)):
		ans = ans + x_1[i]*x_2[i]
	return ans

def kernel_gaussian(x_1,x_2,sigma=1):
	ans = float(0)
	for i in range(len(x_1)):
		ans = ans + (x_1[i]-x_2[i])**2
	ans = math.exp((-1*ans)/(2*sigma*sigma))
	return ans

# Involves some amount of math. Do on paper and then proceed with the code.
def coordinate_ascent_2d(alpha,x_s,y_s,count): # Try to improve this after confirming it Mathematically.

	def h_f(p,q):
		return y_s[p]*y_s[q]*kernel_linear(x_s[p],x_s[q])

	# print(alpha)

	if count == 0:
		return alpha

	a = random.randint(0,len(alpha)-1)
	b = random.randint(0,len(alpha)-1)
	while a == b:
		b = random.randint(0,len(alpha)-1)
	c = float(1) # HYPER-PARAMETER

	k = float(0)
	for i in range(len(alpha)):
		if i == a or i == b:
			continue
		k = k - alpha[i]*y_s[i]

	lhs = 1 - y_s[a]/y_s[b] - (float(k)*y_s[a])/float(c)

	for i in range(len(alpha)):
		if i == a or i == b:
			continue
		lhs = lhs - alpha[i]*(h_f(a,i)-(float(y_s[a]/y_s[b])*h_f(b,i)))

	lhs = lhs - float(k/y_s[b])*h_f(a,b) + k*y_s[a]*h_f(b,b)
	lhs = lhs/(h_f(a,a)-2*(y_s[a]/y_s[b])*h_f(a,b)+h_f(b,b))

	at = lhs
	bt = (k - at*y_s[a])/y_s[b]

	if at >= 0 and bt >= 0:
		alpha[a] = at
		alpha[b] = bt

	return coordinate_ascent_2d(alpha,x_s,y_s,count-1)

def make_prediction(x):

	def f_x_i(i):
		w_t_x = 0
		for j in range(len(alpha)):
			term = alpha[j]*y_s[j]*kernel_linear(x_s[j],x_s[i])
			w_t_x = w_t_x + term
		return w_t_x

	w_t_x = 0
	mx = -1
	mn = -1
	for i in range(len(alpha)):
		term = alpha[i]*y_s[i]*kernel_linear(x_s[i],x)
		w_t_x = w_t_x + term

	for i in range(len(alpha)):
		if y_s[i] == 1:
			temp = f_x_i(i)
			if mn == -1:
				mn = temp
			elif temp < mn:
				mn = temp
		else:
			temp = f_x_i(i)
			if mx == -1:
				mx = temp
			elif temp > mx:
				mx = temp

	b = -(mn+mx)/2
	print("b: ",b)
	if w_t_x + b > 0:
		return 1
	return -1


data_train = get_data('data_train.csv')
data_test = get_data('data_test.csv')
y_s = get_labels(data_train)
data_test_labels = get_labels(data_test)
x_s = data_train

alpha = [0]*(len(x_s)-1)
alpha = coordinate_ascent_2d(alpha,x_s,y_s,100)

correct = 0
for i in range(len(data_test)):
	prediction = make_prediction(data_test[i])
	print("Actual: " + str(data_test_labels[i]) + " Predicted: " + str(prediction))
	if prediction == data_test_labels[i]:
		correct = correct + 1
accuracy = float(correct)/float(len(data_test))
print("Accuracy: " + str(accuracy))