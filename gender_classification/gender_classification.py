from sklearn import tree
from sklearn import neural_network
from sklearn import svm
from sklearn import gaussian_process
from sklearn.metrics import accuracy_score

dt_clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
mlp_clf = neural_network.MLPClassifier()
# 2
svc_clf = svm.SVC()
# 3
gauss_clf =gaussian_process.GaussianProcessClassifier()

classifiers = {'decision_tree' : dt_clf, 'MLP': mlp_clf, 'SVC' : svc_clf, 'gaussian_process' : gauss_clf}

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

X_train = X[0:9]
X_test = X[9:]

Y_train = Y[0:9]
Y_test = Y[9:]
Y_true = ['male', 'male']
result_clfs = dict()
# CHALLENGE - ...and train them on our data
for name in classifiers:
	clf = classifiers[name]
	clf = clf.fit(X_train, Y_train)
	prediction = clf.predict(X_test)
	print(name + ':')
	accuracy = accuracy_score(Y_true, prediction, normalize=False)
	print('prediction result: ' + str(prediction) + '\taccuracy: ' + str(accuracy))
	result_clfs[name] = (accuracy, prediction)

# CHALLENGE compare their reusults and print the best one!
best_accuracy = -1
for name in result_clfs:
	accuracy, prediction = result_clfs[name]
	if accuracy > best_accuracy:
		best_accuracy = accuracy
		best_clf = name
print('The best one: ' + best_clf)
print('prediction result: ' + str(prediction) + '\taccuracy: ' + str(accuracy))