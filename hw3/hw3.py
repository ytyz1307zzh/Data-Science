import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k: int, centroids):
        """
        :param k: number of clusters in k-means
        :param centroids: numpy array of shape (k, features)
        """
        self.k = k
        self.centroids = centroids
        assert len(self.centroids) == self.k

    def fit(self, X, metric="euclidean"):
        """
        :param X: numpy array with shape (N, features)
        :param metric: type of distance metric
        :return Y: predicted labels
        """
        Y = [0 for _ in range(len(X))]  # label assigned to each data
        
        while True:
            num_change = 0
            cls = {cid: [] for cid in range(self.k)}  # data assigned to each cluster

            for xid in range(len(X)):
                x = X[xid]
                dist2cen = []
                for c in self.centroids:
                    if metric == "euclidean":
                        dist = KMeans.Euclidean(x, c)
                    elif metric == "manhattan":
                        dist = KMeans.Manhattan(x, c)
                    else:
                        raise ValueError("Unrecognized distance metric")
                    dist2cen.append(dist)
                cid = np.argmin(dist2cen)  # assign nearest centroid
                
                if cid != Y[xid]:
                    num_change += 1  # the assignment changed or not
                
                # update predictions
                cls[cid].append(x)
                Y[xid] = cid
            
            # converge?
            if num_change == 0:
                break
                
            # update centroids
            for cid in range(self.k):
                self.centroids[cid] = np.average(cls[cid], axis=0)

        return Y

    def SSE(self, X, Y):
        """
        Calculate the sum of square errors
        """
        error = 0.0
        for xid in range(len(X)):
            cid = Y[xid]
            error += KMeans.Euclidean(self.centroids[cid], X[xid]) ** 2
        return error

    @staticmethod
    def Euclidean(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    @staticmethod
    def Manhattan(x, y):
        return np.sum(np.abs(x - y))


# -------------Dr. Jiang's code---------------------
df = pd.read_csv("data-hw3.txt", delimiter="\t")

objects = []
instances = []
for instance in df.to_numpy():
    objects.append(instance[0])
    featureValues = list(instance[1:])
    instances.append(featureValues)
nInstances = len(instances)
print("Number of instances:", nInstances)
print("Data names:")
print(objects)

X = np.array(instances)
print("Data features:")
print(X)
plt.scatter(X[:, 0], X[:, 1])
# -------------Dr. Jiang's code---------------------

# Q1: K=2, Euclidean distance
X1 = X[:, [0, 1]]
KMeans1 = KMeans(k=2, centroids=np.array([(7., 7.), (14., 14.)]))
Y1 = KMeans1.fit(X1, metric="euclidean")
print("\nQ1 (1) prediction: ")
print(Y1)

# visualization
plt.clf()
fig, axs = plt.subplots(2)
colors = ["red" if Y1[xid] == 0 else "blue" for xid in range(len(X))]
axs[0].scatter(X[:, 0], X[:, 1], c=colors)

# another set of initial centroids
X2 = X[:, [0, 1]]
KMeans2 = KMeans(k=2, centroids=np.array([(7., 7.), (7., 14.)]))
Y2 = KMeans2.fit(X2, metric="euclidean")
print("\nQ1 (2) prediction: ")
print(Y2)

# visualization
colors = ["red" if Y2[xid] == 0 else "blue" for xid in range(len(X))]
axs[1].scatter(X[:, 0], X[:, 1], c=colors)
plt.show()

# compare the quality of two clustering
sse1 = KMeans1.SSE(X1, Y1)
sse2 = KMeans2.SSE(X2, Y2)
print(f"SSE: (1) {sse1:.2f}, (2) {sse2:.2f}")

# Q2, k=2, Manhattan distance
X3 = X[:, [2, 3]]
KMeans3 = KMeans(k=2, centroids=np.array([(1., 1.), (25., 25.)]))
Y3 = KMeans3.fit(X3, metric="manhattan")
print("\nQ2 prediction: ")
print(Y3)

# visualization
plt.clf()
colors = ["red" if Y3[xid] == 0 else "blue" for xid in range(len(X))]
plt.scatter(X[:, 2], X[:, 3], c=colors)
plt.show()

# Q3, k=3, Manhattan distance
X4 = X[:, [2, 3]]
KMeans4 = KMeans(k=3, centroids=np.array([(3., 3.), (10., 17.), (23., 10.)]))
Y4 = KMeans4.fit(X4, metric="manhattan")
print("\nQ3 prediction: ")
print(Y4)

# visualization
plt.clf()
colors = []
for xid in range(len(X)):
    if Y4[xid] == 0:
        colors.append("red")
    elif Y4[xid] == 1:
        colors.append("blue")
    else:
        colors.append("green")
plt.scatter(X[:, 2], X[:, 3], c=colors)
plt.show()
