import numpy as np

class_dict = {}
class_dict['cc'] = 0
class_dict['con'] = 1
class_dict['emboss'] = 2
class_dict['jitter'] = 3
class_dict['neg'] = 4
class_dict['noise01'] = 5
class_dict['noise02'] = 6
class_dict['original'] = 7
class_dict['poster'] = 8
class_dict['rot'] = 9
class_dict['smooth'] = 10
class_dict['stipple'] = 11

class_array = [''] * 12
class_array[0] = 'cc'
class_array[1] = 'con'
class_array[2] = 'emboss'
class_array[3] = 'jitter'
class_array[4] = 'neg'
class_array[5] = 'noise01'
class_array[6] = 'noise02'
class_array[7] = 'original'
class_array[8] = 'poster'
class_array[9] = 'rot'
class_array[10] = 'smooth'
class_array[11] = 'stipple'

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, task1=False):
        self.max_depth = max_depth
        self.task1 = task1

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        predictions = []
        for inputs in X:
            node = self.tree_
            i = 0
            while node.left:
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
                i = i + 1
            predictions.append(node.predicted_class)
        return predictions

    def _gini(self, y):
        m = len(y)
        if (self.task1):
            return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in class_array)
        else:
            return 1.0 - sum((np.sum(y == str(c+1)) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        m = len(y)
        if m <= 1:
            return None, None

        if (self.task1):
            num_parent = [np.sum(y == c) for c in class_array]
        else:
            num_parent = [np.sum(y == str(c+1)) for c in range(self.n_classes_)]

        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                if (self.task1):
                    num_left[class_dict[c]] += 1
                    num_right[class_dict[c]] -= 1
                else:
                    num_left[int(c)-1] += 1
                    num_right[int(c)-1] -= 1                    
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        if (self.task1):
            num_samples_per_class = [np.sum(y == i) for i in class_array]
        else:
            num_samples_per_class = [np.sum(y == str(i+1)) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr

                X_left = X[indices_left]
                y_left = y[indices_left]
                X_right = X[~indices_left]
                y_right = y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node


