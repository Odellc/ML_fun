#Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#import laptop train data
laptop_data = pd.read_csv("./data/laptops_train.csv")


X = laptop_data["Price"].values
y = laptop_data["RAM"].values

#reshape
X = X.reshape(-1, 1)

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)


#Fit the classifier to the training data
knn.fit(X_train, y_train)


#print the accuracy
print(knn.score(X_test, y_test))


#Create neighbors
neighbors = np.arange(1, 13)

train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:

    #Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    #Fit the model
    knn.fit(X_train, y_train)

    #Comput accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print( neighbors, '\n', train_accuracies, '\n', test_accuracies)


# Add a title
plt.title("KNN: Varying Number of Neighbors")

#Plot training accuracries
plt.plot( neighbors, train_accuracies.values(), label="Training Accuracy")

#Plot test accuracries
plt.plot( neighbors , test_accuracies.values(), label="Test Accuracy")

plt.legend()
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")

#Display the plot
plt.show()