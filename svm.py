import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score  

# loading the dataset
dataset = pd.read_csv('ovbarve.csv')
X = dataset.iloc[:, [0, 1]].values
labels = dataset.iloc[:, [2]].values
#reshape labels
labels = np.ravel(labels)

#Set path to results folder to store the results
os.chdir("results")

#Read data 2-dimensional data into individual array X1 and X2 for scatter plot 
X1 = np.ravel(dataset.iloc[:, [0]].values)
X2 = np.ravel(dataset.iloc[:, [1]].values)

#Scatter plot data with labels colored seperately
color= ['red' if label == 1 else 'green' for label in labels]
plt.scatter(X1, X2, color=color)
plt.title("Scatter plot of the data for given Label")
plt.xlabel("x1 dimension")
plt.ylabel("x2 dimension")
plt.savefig("Scatterplot.png")
plt.show()

# Scale each feature between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

#Spit data into 5 folds
kf = KFold(n_splits=5, shuffle=False)
C = [2 ** i for i in range(-5,15,2)]
gamma = [2 ** i for i in range(-15,3,2)]


best_value_C = 0
best_value_gamma = 0
highest_accuracy = 0
accuracy_array_for_grid_search = []
for c_ in C:
    for gamma_ in gamma:
        accuracy = []        
        #For each fold iteration (So 5 iterations)
        for train_index, test_index in kf.split(X_scaled):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            #SVM clustering
            clf = svm.SVC(C=c_, gamma=gamma_)
            clf.fit(X_train, y_train)
            #predict
            predicted = clf.predict(X_test)
            acc = accuracy_score(y_test, predicted)             
            accuracy.append(acc)
        #print(accuracy)
        accuracy_ = np.mean(accuracy)
        if accuracy_> highest_accuracy:
            best_value_C = c_
            best_value_gamma = gamma_
            highest_accuracy = accuracy_
        accuracy_array_for_grid_search.append(accuracy_)

fig = plt.figure()
ax1 = Axes3D(fig)


print("BEST values of C and Gamma initial search over wider range")
print("C:", best_value_C,  "Gamma:", best_value_gamma, "Accuracy:", highest_accuracy)

# 3d bar plot
_xx, _yy = np.meshgrid(C, gamma)
x, y = _xx.ravel(), _yy.ravel()
bottom = np.zeros_like(accuracy_array_for_grid_search)
width = depth = 0.01
ax1.bar3d(x, y, bottom, width, depth, accuracy_array_for_grid_search, shade=True, color='b')
ax1.set_title('Bar plot for grid search')
ax1.set_xlabel("C")
ax1.set_ylabel("Gamma")
ax1.set_zlabel("Accuracy")
plt.savefig("3d_bar_intial_search.png")
plt.show()

#3d scatter plot
fig = pyplot.figure()
ax = Axes3D(fig)
p = ax.scatter(x, y, accuracy_array_for_grid_search, c=accuracy_array_for_grid_search, cmap=plt.cm.cool)
fig.colorbar(p)
plt.title("Scatter plot for grid search")
ax.set_xlabel("C")
ax.set_ylabel("Gamma")
ax.set_zlabel("Accuracy")
plt.savefig("3d_scatter_intial_search.png")
pyplot.show()


data = np.array(accuracy_array_for_grid_search)
shape = (9, 10)
newdata = data.reshape(shape)

fig, ax = plt.subplots()
plt.imshow(newdata, interpolation="None")
plt.colorbar()
ax.set_xticks(np.arange(len(C)))
ax.set_yticks(np.arange(len(gamma)))
# ... and label them with the respective list entries
ax.set_xticklabels(C)
ax.set_yticklabels(gamma)

for i in range(len(newdata)):
    for j in range(len(newdata[0])):
        text = ax.text(j, i, format(newdata[i, j], ".2f"),
                       ha="center", va="center", color="w")

ax.set_title("Grid Search heatmap")
fig.tight_layout()
plt.xlabel("C")
plt.ylabel("Gamma")
plt.savefig("heatmap_intial_search.png")
plt.show()



#FINER SEARCH OVER C AND GAMMA PARAMETERS

gamma= [0.125+(i/10 )for i in range(9)]
C = [1.5 + i/10 for i in range(10)]

best_value_C = 0
best_value_gamma = 0
highest_accuracy = 0
accuracy_array_for_grid_search = []
for c_ in C:
    for gamma_ in gamma:
        accuracy = []
        #For each fold iteration (So 5 iterations)
        for train_index, test_index in kf.split(X_scaled):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            #SVM clustering
            clf = svm.SVC(C=c_, gamma=gamma_)
            clf.fit(X_train, y_train)
            #predict
            predicted = clf.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            accuracy.append(acc)
        #print(accuracy)
        accuracy_ = np.mean(accuracy)
        if accuracy_> highest_accuracy:
            best_value_C = c_
            best_value_gamma = gamma_
            highest_accuracy = accuracy_
        accuracy_array_for_grid_search.append(accuracy_)

fig = plt.figure()
ax1 = Axes3D(fig)

print("BEST values of C and Gamma after fine tuning the range")
print("C:", best_value_C,  "Gamma:", best_value_gamma,"Accuracy:", highest_accuracy)

# 3d bar plot
_xx, _yy = np.meshgrid(C, gamma)
x, y = _xx.ravel(), _yy.ravel()
bottom = np.zeros_like(accuracy_array_for_grid_search)
width = depth = 0.01
ax1.bar3d(x, y, bottom, width, depth, accuracy_array_for_grid_search, shade=True, color='b')
ax1.set_title('Bar plot for grid search')
ax1.set_xlabel("C")
ax1.set_ylabel("Gamma")
ax1.set_zlabel("Accuracy")
plt.savefig("3d_bar_fine_grain_search.png")
plt.show()

#3d scatter plot
fig = pyplot.figure()
ax = Axes3D(fig)
p = ax.scatter(x, y, accuracy_array_for_grid_search, c=accuracy_array_for_grid_search, cmap=plt.cm.cool)
fig.colorbar(p)
plt.title("Scatter plot for grid search")
ax.set_xlabel("C")
ax.set_ylabel("Gamma")
ax.set_zlabel("Accuracy")
plt.savefig("3d_scatter_fine_grain_search.png")
pyplot.show()


data = np.array(accuracy_array_for_grid_search)
shape = (9, 10)
newdata = data.reshape(shape)

fig, ax = plt.subplots()
plt.imshow(newdata, interpolation="None")
plt.colorbar()
ax.set_xticks(np.arange(len(C)))
ax.set_yticks(np.arange(len(gamma)))
# ... and label them with the respective list entries
ax.set_xticklabels(C)
ax.set_yticklabels(gamma)

for i in range(len(newdata)):
    for j in range(len(newdata[0])):
        text = ax.text(j, i, format(newdata[i, j], ".2f"),
                       ha="center", va="center", color="w")

ax.set_title("Grid Search heatmap")
fig.tight_layout()
plt.xlabel("C")
plt.ylabel("Gamma")
plt.savefig("heatmap_fine_grain_search.png")
plt.show()
























