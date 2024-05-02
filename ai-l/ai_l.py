import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1,
                           n_informative=2, n_redundant=0, n_repeated=0, random_state=42)

# Initialize the classifier
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=100, random_state=42)

# Create a scatter plot to visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=100, edgecolors='k')
plt.title('Initial Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Initialize the plot for visualization during training
plt.ion()
plt.show()

# Train the classifier and visualize the decision boundary in real-time
for i in range(100):
    clf.partial_fit(X, y, classes=np.unique(y))
    
    # Plot decision boundary
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=100, edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.title('Classifier Learning (Iteration {})'.format(i+1))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.pause(0.1)

plt.ioff()
plt.show()
