##### Problem 2f #####
for i in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
    LASSO(imDims, measurements, X, i)
    print('lambda:', i)
    plt.show()

##### Problem 3 #####
def LS(U, y):
    """
    Input: U is an n X d matrix with orthonormal columns and y is an n X 1 vector.
    Output: The OLS estimate what_{LS}, a d X 1 vector.
    """
    wls = np.dot(U.T, y) #pseudoinverse of orthonormal matrix is its transpose
    return wls


def thresh(U, y, lmbda):
    """
    Input: U is an n X d matrix and y is an n X 1 vector; lambda is a scalar threshold of the entries.
    Output: The estimate what_{T}(lambda), a d X 1 vector that is hard-thresholded (in absolute value) at level lambda.
            When code is unfilled, returns the all-zero d-vector.
    """
    n, d = np.shape(U)
    wls = LS(U, y)
    what = np.zeros(d)
    
    #print np.shape(wls)
    ##########
    #TODO: Fill in thresholding function; store result in what
    #####################
    #YOUR CODE HERE:   
    wls[np.abs(wls) < lmbda] = 0
    what = wls    
    ###############
    return what
    
    
def topk(U, y, s):
    """
    Input: U is an n X d matrix and y is an n X 1 vector; s is a positive integer.
    Output: The estimate what_{top}(s), a d X 1 vector that has at most s non-zero entries.
            When code is unfilled, returns the all-zero d-vector.
    """
    n, d = np.shape(U)
    what = np.zeros(d)
    wls = LS(U, y)
    
    ##########
    #TODO: Fill in thresholding function; store result in what
    #####################
    #YOUR CODE HERE: Remember the absolute value!
    wls[np.argpartition(np.abs(wls), d-s)[:d-s]] = 0
    what = wls
    ###############
    return what


#nrange contains the range of n used, ls_error the corresponding errors for the OLS estimate
nrange, ls_error, _, _ = error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=True, true_s=5)

########
#TODO: Your code here: call the helper function for d and s, and plot everything
########
#YOUR CODE HERE:
drange, ls_errord, _, _ = error_calc(num_iters=10, param='d', n=1000, d=100, s=5, s_model=True, true_s=5)
srange, ls_errors, _, _ = error_calc(num_iters=10, param='s', n=1000, d=100, s=5, s_model=True, true_s=5)
plt.plot(np.log(nrange), np.log(ls_error))
plt.xlabel('Log(n)')
plt.ylabel('Log(error)')
plt.show()
plt.plot(drange, ls_errord)
plt.xlabel('d')
plt.ylabel('Error')
plt.show()
plt.plot(srange, ls_errors)
plt.xlabel('s')
plt.ylabel('Error')
plt.ylim((0, 1))
plt.show()

#TODO: Part (b)
##############
#YOUR CODE HERE:
nrange, ls_error, topk_error, thresh_error = error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=True)
plt.plot(np.log(nrange), np.log(ls_error), label='LS')
plt.plot(np.log(nrange), np.log(topk_error), label='Topk')
plt.plot(np.log(nrange), np.log(thresh_error), label='Thresh')
plt.xlabel('Log(n)')
plt.ylabel('Log(error)')
plt.legend()
plt.show()

drange, ls_errord, topk_errord, thresh_errord = error_calc(num_iters=10, param='d', n=1000, d=100, s=5, s_model=True)
plt.plot(drange, ls_errord, label='LS')
plt.plot(drange, topk_errord, label='Topk')
plt.plot(drange, thresh_errord, label='Thresh')
plt.xlabel('d')
plt.ylabel('Error')
plt.legend()
plt.show()

srange, ls_errors, topk_errors, thresh_errors = error_calc(num_iters=10, param='s', n=1000, d=100, s=5, s_model=True)
plt.plot(srange, ls_errors, label='LS')
plt.plot(srange, topk_errors, label='Topk')
plt.plot(srange, thresh_errors, label='Thresh')
plt.xlabel('d')
plt.ylabel('Error')
plt.legend()
plt.show()

#TODO: Part (c)
##############
#YOUR CODE HERE:
nrange, ls_error, topk_error, thresh_error = error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=False, true_s=30)
plt.plot(np.log(nrange), np.log(ls_error), label='LS')
plt.plot(np.log(nrange), np.log(topk_error), label='Topk')
plt.plot(np.log(nrange), np.log(thresh_error), label='Thresh')
plt.xlabel('Log(n)')
plt.ylabel('Log(error)')
plt.legend()
plt.show()

drange, ls_errord, topk_errord, thresh_errord = error_calc(num_iters=10, param='d', n=1000, d=100, s=5, s_model=False, true_s=30)
plt.plot(drange, ls_errord, label='LS')
plt.plot(drange, topk_errord, label='Topk')
plt.plot(drange, thresh_errord, label='Thresh')
plt.xlabel('d')
plt.ylabel('Error')
plt.legend()
plt.show()

srange, ls_errors, topk_errors, thresh_errors = error_calc(num_iters=10, param='s', n=1000, d=100, s=5, s_model=False, true_s=30)
plt.plot(srange, ls_errors, label='LS')
plt.plot(srange, topk_errors, label='Topk')
plt.plot(srange, thresh_errors, label='Thresh')
plt.xlabel('d')
plt.ylabel('Error')
plt.legend()
plt.show()


##### Problem 4 #####
import io
from collections import Counter
from pydot import graph_from_dot_data

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

import pydot

eps = 1e-5  # a small number

def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (
                self.features[self.split_idx], self.thresh,
                self.left.__repr__(), self.right.__repr__())


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        
    @staticmethod
    def entropy(y):
        if y.size == 0:
            return 0
        p = np.where(y < 0.5)[0].size / y.size
        if np.abs(p) < 1e-8 or np.abs(1-p) < 1e-8:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
            
    @staticmethod
    def information_gain(X, y, thresh):
        base = DecisionTree.entropy(y)
        y0 = y[np.where(X < thresh)[0]]
        p0 = y0.size / y.size
        y1 = y[np.where(X >= thresh)[0]]
        p1 = y1.size / y.size
        en = p0 * DecisionTree.entropy(y0) + p1 * DecisionTree.entropy(y1)
        return base - en
    
    @staticmethod
    def gini(y):
        if y.size == 0:
            return 0
        p = np.where(y < 0.5)[0].size / y.size
        return 1 - p ** 2 - (1-p) **2

    @staticmethod
    def gini_impurity(X, y, thresh):
        base = DecisionTree.gini(y)
        y0 = y[np.where(X < thresh)[0]]
        p0 = y0.size / y.size
        y1 = y[np.where(X >= thresh)[0]]
        p1 = y1.size / y.size
        gi = p0 * DecisionTree.gini(y0) + p1 * DecisionTree.gini(y1)
        return base - gi

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        for i in range(self.n):
            idx = np.random.randint(0, X.shape[0], X.shape[0])
            X_new, y_new = X[idx, :], y[idx]
            self.decision_trees[i].fit(X_new, y_new)
        return self

    def predict(self, X):
        yhat = [self.decision_trees[i].predict(X) for i in range(self.n)]
        return np.array(np.round(np.mean(yhat, axis=0)), dtype=np.bool)

class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        i = 0
        while i < self.n:
            idx = np.random.choice(X.shape[0], size=X.shape[0], p=self.w)
            X_new, y_new = X[idx, :], y[idx]
            self.decision_trees[i].fit(X_new, y_new)
            wrong = np.abs((y - self.decision_trees[i].predict(X)))
            error = wrong.dot(self.w) / np.sum(self.w)
            self.a[i] = 0.5 * np.log2((1 - error) / error)
            idx_wrong = np.where(wrong > 0.5)[0]
            idx_correct = np.where(wrong <= 0.5)[0]
            self.w[idx_wrong] = self.w[idx_wrong] * np.exp(self.a[i])
            self.w[idx_correct] = self.w[idx_correct] * np.exp(-self.a[i])
            self.w /= np.sum(self.w)
            i += 1
        return self

    def predict(self, X):
        yhat = [self.decision_trees[i].predict(X) for i in range(self.n)]
        p0 = self.a.dot(np.array(yhat) == 0)
        p1 = self.a.dot(np.array(yhat) == 1)
        return np.array(np.argmax(np.vstack([p0, p1]), axis=0), dtype=np.bool)


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)


if __name__ == "__main__":
    #dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)
    print("Predictions", dt.predict(Z)[:100])

    print("\n\nPart (c): sklearn's decision tree")
    clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    sklearn.tree.export_graphviz(clf, out_file=out, feature_names=features, class_names=class_names)
    graph = pydot.graph_from_dot_data(out.getvalue())
    pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # Bagged trees
    print("\n\nPart (d-e): bagged trees")
    bt = BaggedTrees(params, n=N)
    bt.fit(X, y)
    evaluate(bt)
    print(repr(bt))

    # Random forest
    print("\n\nPart (f-g): random forest")
    rf = RandomForest(params, n=N, m=np.int(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    evaluate(rf)

    # Boosted random forest
    print("\n\nPart (h): boosted random forest")
    boosted = BoostedRandomForest(params, n=N, m=np.int(np.sqrt(X.shape[1])))
    boosted.fit(X, y)
    evaluate(boosted)