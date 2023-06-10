import math
import numpy as np
from scipy import spatial
class PPRClassifier:
    def __init__(self, d):
        self.d = d

    def addTrainingData(self, trainingData, trainingLabels):
        self.data = trainingData
        self.labels_list = trainingLabels
        self.labels_counts = {}

        for lbl in set(trainingLabels):
            self.labels_counts[lbl] = trainingLabels.count(lbl)

    def addTestingData(self, testingData):
        self.testingDataStartIndex = len(self.data)
        self.data = np.concatenate((self.data, testingData), axis=0)


    # Use Cosine Similarity for creating graph
    def setupGraph(self):
        graph = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data)):
                sim = math.pow(math.e, -(np.linalg.norm(self.data[i]-self.data[j]))**2)
                row.append(sim)
                # if i == j:
                #     # Removing connections leading to same node
                #     row.append(0)
                # else:
                #     similarity_score = 1 - spatial.distance.cosine(self.data[i], self.data[j])
                #     row.append(similarity_score)
            row = np.asarray(row)/np.linalg.norm(np.asarray(row), ord=1)
            graph.append(row)

        self.graph = np.asarray(graph).T
        # print(self.graph[0])
    # Perform prediction on all data in the testing set, outputs the predictions to a list
    def predict(self):
        predictions = []

        random_walks = []
        label_order = []

        first_term = (1-self.d)*np.linalg.inv(np.eye(len(self.data))-self.d*self.graph)
        for c in self.labels_counts.keys():
            label_order.append(c)
            # Create u vector, set to 1 if Y^L_i = c
            u = []
            for lbl in self.labels_list:
                if lbl == c:
                    u.append(1)
                else:
                    u.append(0)

            for i in range(self.testingDataStartIndex, len(self.data)):
                u.append(0)

            u = np.asarray(u)
            # Normalize
            u = u/np.linalg.norm(u,ord=1)
            # random_walks.append(self.randomWalk(u))
            random_walks.append(np.dot(first_term, u))


            self.rw = random_walks

        for i in range(self.testingDataStartIndex, len(self.data)):
            r_vals = np.asarray([r[i] for r in random_walks])
            # print(r_vals)
            # print(np.argmax(r_vals))
            predictions.append(label_order[np.argmax(r_vals)])
            # print(r_vals[np.argmax(r_vals)])

        return predictions

    def randomWalk(self, u):
        # inn = np.dot(np.linalg.inv(np.eye(len(self.graph)) - (self.d * self.graph)),u)
        # print(inn.shape)
        # eigen_vals, eigen_vecs = np.linalg.eig((1-self.d)* np.dot(
        #     np.linalg.inv(np.eye(len(self.graph))-(self.d*self.graph)),
        #     u))
        #
        # eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        #
        # eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        # return eigen_pairs[0][:,0]

        # r = np.random.rand(len(self.data))
        r = np.dot(np.linalg.inv(np.eye(len(self.data))-self.d*self.graph),(1-self.d)*u)
        # r = np.full(len(self.data), 1/len(self.data))
        # for i in range(5000):
        #     prev_r = r.copy()
        #     # Update r
        #     r = (1-self.d)*u + self.d*np.dot(self.graph,r)
        #
        #     # Check for convergence
        #     if np.allclose(prev_r, r):
        #         print('Stopping early at ' + str(i) + ' iterations')
        #         break

        return r