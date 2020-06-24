#Final Project
#INF 552
#Shagun Gupta
#Lisa Meng

#Importing data to google colab

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import math
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import graphviz
drive.mount('/content/drive')
default_path = '/content/drive/My Drive/INF ML 552/final project/'

data = pd.read_csv(default_path + 'data.csv')
data["diagnosis"] = data["diagnosis"].replace('B',0)
data["diagnosis"] = data["diagnosis"].replace('M',1)

training_data = pd.DataFrame(columns=data.columns)
testing_data = pd.DataFrame(columns=data.columns)
# rows = list(range(data.shape[0]))
rows = [154, 508, 129, 162, 59, 71, 49, 164, 534, 96, 143, 104, 147, 348, 88, 21, 103, 396, 194, 520, 562, 484, 252, 179, 488, 527, 395, 73, 416, 367, 187, 371, 385, 153, 4, 424, 548, 251, 98, 338, 435, 426, 281, 411, 485, 473, 567, 171, 436, 528, 254, 170, 113, 112, 343, 309, 346, 336, 464, 248, 107, 284, 498, 77, 230, 400, 237, 452, 126, 505, 138, 69, 16, 308, 146, 347, 321, 489, 398, 165, 55, 365, 276, 144, 479, 135, 41, 421, 547, 486, 75, 220, 372, 442, 427, 70, 238, 492, 501, 519, 461, 402, 82, 494, 333, 10, 522, 90, 383, 553, 6, 335, 495, 243, 89, 93, 124, 289, 3, 137, 359, 483, 440, 64, 221, 132, 169, 307, 315, 515, 532, 551, 463, 205, 241, 67, 155, 56, 87, 476, 560, 295, 280, 40, 529, 325, 454, 399, 313, 550, 256, 100, 406, 160, 503, 47, 302, 130, 283, 327, 203, 523, 384, 554, 36, 382, 340, 349, 273, 211, 134, 351, 265, 331, 199, 540, 234, 0, 260, 370, 32, 244, 232, 78, 478, 91, 117, 255, 195, 470, 450, 558, 493, 74, 239, 557, 208, 122, 148, 456, 306, 106, 407, 369, 43, 2, 201, 163, 536, 535, 29, 439, 139, 288, 11, 48, 46, 300, 142, 25, 222, 354, 480, 23, 274, 363, 17, 303, 8, 52, 417, 366, 190, 357, 506, 513, 556, 262, 469, 7, 189, 50, 509, 433, 5, 420, 392, 86, 471, 410, 34, 430, 312, 30, 19, 568, 294, 176, 114, 152, 216, 546, 178, 196, 543, 119, 271, 487, 99, 446, 323, 362, 455, 412, 246, 415, 145, 183, 332, 364, 225, 563, 376, 292, 474, 413, 250, 504, 388, 290, 287, 54, 63, 342, 453, 109, 314, 350, 133, 353, 318, 341, 279, 511, 223, 95, 566, 482, 544, 233, 28, 374, 97, 128, 389, 539, 118, 360, 58, 264, 218, 257, 329, 168, 184, 202, 110, 83, 14, 13, 240, 459, 437, 330, 213, 84, 37, 277, 166, 81, 188, 57, 393, 186, 449, 27, 172, 428, 356, 305, 391, 120, 105, 266, 500, 339, 12, 344, 209, 156, 408, 381, 564, 431, 53, 419, 72, 328, 200, 275, 60, 61, 518, 434, 35, 226, 394, 541, 324, 397, 530, 497, 552, 206, 197, 18, 531, 537, 20, 38, 378, 80, 565, 245, 432, 161, 1, 204, 65, 311, 68, 316, 272, 242, 227, 425, 451, 319, 538, 358, 317, 561, 85, 214, 44, 491, 15, 173, 33, 299, 510, 334, 297, 123, 175, 151, 219, 259, 296, 131, 443, 229, 414, 418, 429, 386, 441, 499, 345, 263, 215, 477, 526, 269, 445, 9, 545, 403, 66, 404, 507, 136, 368, 380, 291, 352, 298, 121, 217, 337, 301, 39, 310, 212, 444, 460, 447, 502, 521, 458, 92, 285, 108, 517, 462, 514, 293, 193, 149, 542, 62, 31, 401, 140, 375, 475, 278, 559, 159, 286, 267, 457, 467, 127, 468, 512, 236, 185, 198, 387, 224, 361, 192, 102, 177, 490, 472, 253, 438, 158, 320, 141, 207, 268, 304, 465, 249, 174, 45, 448, 379, 533, 24, 182, 549, 42, 79, 373, 247, 115, 258, 181, 191, 180, 125, 555, 282, 51, 390, 466, 270, 116, 409, 355, 235, 524, 76, 22, 228, 377, 326, 111, 261, 496, 516, 150, 231, 167, 101, 210, 157, 26, 525, 322, 423, 422, 94, 481, 405]
random.shuffle(rows)
for index in rows[0:int(len(rows)*0.7)+1]:
  training_data = training_data.append(pd.DataFrame(data.iloc[index]).transpose())

for index in rows[int(len(rows)*0.7)+1:]:
  testing_data = testing_data.append(pd.DataFrame(data.iloc[index]).transpose())

X_train = training_data.drop(columns={"diagnosis"})
X_train = X_train[["radius_mean", "perimeter_mean", "area_mean", "texture_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"]]
X_train = X_train.reset_index(drop=True)
y_train = pd.DataFrame(training_data["diagnosis"])
y_train = y_train.reset_index(drop=True)

X_test = testing_data.drop(columns={"diagnosis"})
X_test = X_test[["radius_mean", "perimeter_mean", "area_mean", "texture_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"]]
X_test = X_test.reset_index(drop=True)
y_test = pd.DataFrame(testing_data["diagnosis"])
y_test = y_test.reset_index(drop=True)

scale = StandardScaler(copy=True, with_mean=True, with_std=True)
scale.fit(X_train)

X_train_std = scale.transform(X_train)
X_test_std = scale.transform(X_test)

#Neural Networks

def get_weights(n):
    w_ij = []
    for i in range(n):
        i = np.random.uniform(-0.1, 0.1)
        w_ij.append(i)
    return w_ij


def sigmoid(s):
    # print(s)
    return 1 / (1 + np.exp(-s))


def deriv_sigmoid(s):
    return sigmoid(s) * (1 - sigmoid(s))


def feed_forward(w, input_x):
    x = []
    # print(input_x)
    x.append(list(input_x))
    x[0].insert(0, 1)
    x.append([0] * hidden_layer_size)
    # x[1].insert(0,1)
    x.append([0] * output_size)

    for l in range(1, layers):
        for j in range(level[l]):
            for i in range(len(x[l - 1])):
                x[l][j] += w[l - 1][i][j] * x[l - 1][i]
            x[l][j] = sigmoid(x[l][j])
        x[l].insert(0, 1)

    x[2] = x[2][1:]
    return x


def get_delta(x, w, y):
    delta = [[0]]
    delta[0][0] = 2 * (x[2][0] - y) * (1 - math.pow(x[2][0], 2))
    delta.insert(0, [0] * hidden_layer_size)
    for i in range(1, hidden_layer_size):
        delta[0][i] = 1 - math.pow(x[1][i + 1], 2) * w[1][i + 1][0] * delta[1][0]
    return delta


def step3_weights(x, w, delta, lr):
    for l in range(0, layers - 1):
        for j in range(len(delta[l])):
            for i in range(len(x[l])):
                w[l][i][j] -= lr * x[l][i] * delta[l][j]
    return w


# weights2 = step3_weights(lala,weights,delta, learning_rate)

def ffbpnn(w, lr, train_X, train_y):
    ep = 0
    for a in range(int(epochs / N)):
        for i in range(N):
            # print(ep)
            x = feed_forward(w, train_X.iloc[i])
            delta = get_delta(x, w, train_y.iloc[i][0])
            w = step3_weights(x, w, delta, lr)
            ep += 1

    for i in range(epochs - (int(epochs / N) * N)):
        # print(ep)
        x = feed_forward(w, train_X.iloc[i])
        delta = get_delta(x, w, train_y.iloc[i][0])
        w = step3_weights(x, w, delta, lr)
        ep += 1

    return w


for i in range(20):
    print(i)
    # 1 hidden layer with 3 neurons
    learning_rate = 1
    epochs = 2000
    N = X_train.shape[0]
    input_size = X_train.shape[1]
    output_size = 1
    hidden_layer_size = 3
    layers = 3
    weights = [[], []]
    weights[0] = [get_weights(hidden_layer_size) for i in range(input_size + 1)]
    weights[1] = [get_weights(output_size) for i in range(hidden_layer_size + 1)]
    level = [input_size, hidden_layer_size, output_size]
    w = ffbpnn(weights, learning_rate, X_train, y_train)
    #
    #     # after normalization

    N = X_train_std.shape[0]
    input_size = X_train_std.shape[1]
    layers = 3
    weights = [[], []]
    weights[0] = [get_weights(hidden_layer_size) for i in range(input_size + 1)]
    weights[1] = [get_weights(output_size) for i in range(hidden_layer_size + 1)]
    level = [input_size, hidden_layer_size, output_size]
    w = ffbpnn(weights, learning_rate, pd.DataFrame(X_train_std), y_train)


    def testing(test_X, weights):
        y_pred = test_X.apply(lambda z: feed_forward(weights, z)[2][0], axis=1)
        return y_pred


    y_pred = testing(pd.DataFrame(X_train_std), w)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(columns={0: 'prediction'})
    y_pred['label'] = y_train["diagnosis"]
    y_pred['final'] = y_pred['prediction'].apply(lambda z: (0 if z < 0.5 else 1))
    y_pred['compare'] = y_pred['label'] == y_pred['final']
    print("training norm ", sum(y_pred['compare']) / y_pred.shape[0])

    y_pred = testing(pd.DataFrame(X_test_std), w)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(columns={0: 'prediction'})
    y_pred['label'] = y_test["diagnosis"]
    y_pred['final'] = y_pred['prediction'].apply(lambda z: (0 if z < 0.5 else 1))
    y_pred['compare'] = y_pred['label'] == y_pred['final']
    print("testing norm ", sum(y_pred['compare']) / y_pred.shape[0])

mlp = MLPClassifier(hidden_layer_sizes=(3), activation='logistic', solver='sgd',
                    learning_rate='constant', learning_rate_init=1, max_iter=2000)
mlp.fit(X_train_std, y_train.values.ravel())
predictions = mlp.predict(X_test_std)


print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
n_foldCV = 10
scores = cross_val_score(mlp, X_test_std, y_test, cv=n_foldCV)
# print("Scores:")
# [print(score) for score in scores]
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


##PCA + SVM

import numpy as np
import cvxopt
from sklearn.svm import SVC

y_train["diagnosis"] = y_train["diagnosis"].replace(0,-1)

y_test["diagnosis"] = y_test["diagnosis"].replace(0,-1)

def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)

    return f


class SupportVectorMachine(object):
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(power=self.power, gamma=self.gamma, coef=self.coef)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:  # if its empty
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        #     # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        #     # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-11
        # print(idx)
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[i] * self.kernel(
                self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            # print(sample)
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.kernel(
                    self.support_vectors[i], sample)
                # print(self.lagr_multipliers[i],self.support_vector_labels[i],self.kernel(self.support_vectors[i], sample))
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)


def accuracy_score(y_test, y_predict):
    acc = 0
    for i in range(len(y_test)):
        if y_test[i][0] == y_pred[i][0]:
            acc += 1

    return acc / len(y_test)


gamma = 0.065
c = 2

# without PCA

clf = SupportVectorMachine(kernel=rbf_kernel, gamma=gamma, C=c)
clf.fit(X_train_std, np.array(y_train))
# print(clf.support_vector_labels)
y_pred = clf.predict(X_test_std)
# print(y_pred)
accuracy = accuracy_score(np.array(y_test), y_pred)
print("Accuracy std (scratch):", accuracy)

clf_sklearn = SVC(gamma=gamma, kernel='rbf', C=c)
clf_sklearn.fit(X_train_std, y_train["diagnosis"])
y_pred2 = clf_sklearn.predict(X_test_std)
accuracy = accuracy_score(np.array(y_test), y_pred2)
print("Accuracy :", accuracy)

# with PCA

for i in range(1, 8):
    print("Components ", i)

    pca = PCA(n_components=i, svd_solver="randomized")
    pca.fit(X_train_std)
    X_train_pca = pca.transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    clf = SupportVectorMachine(kernel=rbf_kernel, gamma=gamma, C=c)
    clf.fit(np.array(X_train_pca), np.array(y_train))
    # print(clf.support_vector_labels)
    y_pred = clf.predict(np.array(X_test_pca))
    # print(y_pred)
    accuracy = accuracy_score(np.array(y_test), y_pred)
    print("Accuracy (scratch):", accuracy)

    clf_sklearn = SVC(gamma=gamma, kernel='rbf', C=c)
    clf_sklearn.fit(X_train_pca, y_train["diagnosis"])
    y_pred2 = clf_sklearn.predict(X_test_pca)
    accuracy = accuracy_score(np.array(y_test), y_pred2)
    print("Accuracy :", accuracy)


#Decision Trees

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)
dot_data = tree.export_graphviz(clf, out_file=None,max_depth=5,
    feature_names=X_train.columns,class_names=['B',"M"],label='all',
    filled=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph

predictions=clf.predict(X_test)
print(predictions)

n_foldCV=10
scores = cross_val_score(clf, X_test, y_test, cv=n_foldCV)
print("Scores:")
[print(score) for score in scores]
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

td = training_data.drop(data.columns[-1], axis=1)
td = td.drop(columns={"id"})
td = td[["radius_mean", "perimeter_mean", "area_mean", "texture_mean", "smoothness_mean", "compactness_mean",
         "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "diagnosis"]]

ttd = testing_data[["radius_mean", "perimeter_mean", "area_mean", "texture_mean", "smoothness_mean", "compactness_mean",
                    "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 'diagnosis']]
testing_datatable = []
for i in range(len(ttd)):
    row = list(ttd.iloc[i, :])
    testing_datatable.append(row)

datatable = []
for i in range(len(td)):
    row = list(td.iloc[i, :])
    datatable.append(row)


def label_count(df):
    counts = {}
    if type(df) == pd.core.frame.DataFrame:
        label = df["diagnosis"]
        for i in label:
            if i not in counts:
                counts[i] = 0
            counts[i] += 1
    else:
        for row in df:
            # in our dataset format, the label is in the last column
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
    return counts


header = td.columns


class Question:

    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

    def match(self, example):
        val = example[self.attribute]
        return val <= self.value

    def __repr__(self):
        condition = "<="
        return "Is %s %s %s?" % (
            header[self.attribute], condition, float(self.value))


def partition(data, question):
    """Partitions dataset

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    #     for i in range(len(data)):
    #         row = list(data.iloc[i, : ])
    for row in data:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def entropy(dt):
    counts = label_count(dt)
    uncertainty = 0
    for label in counts:
        p = counts[label] / (len(dt))
        uncertainty += -1 * (p * math.log2(p))
    return uncertainty


def info_gain(root_uncertainty, true, false):
    p = float(len(true)) / (len(true) + len(false))
    return root_uncertainty - (p * entropy(true)) - ((1 - p) * entropy(false))  # avg entropy


def find_root_node(dt):
    best_gain = 0  # keep track of best info gain
    best_question = None  # keep track of feature that produced it
    current_uncertainty = entropy(dt)
    n_features = len(dt[0]) - 1

    #     datatable=[]
    #     for i in range(len(dt)):
    #         row = list(dt.iloc[i, : ])
    #         datatable.append(row)
    for col in range(n_features):
        values = set([r[col] for r in dt])

        for val in values:
            question = Question(col, val)

            # splitting dataset
            true_rows, false_rows = partition(dt, question)

            # Skip this split if it doesn't divide the dataset
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate info gain from this split
            gain = info_gain(current_uncertainty, true_rows, false_rows)

            if gain > best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:

    def __init__(self, dt):
        self.predictions = label_count(dt)


class Decision_Node:

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(dt):
    # partition data on each unique attribute
    # calculate the info gain
    # return question that produces highest gain
    gain, question = find_root_node(dt)

    # Base case: info gain is 0 --> no further questions to ask --> return leaf
    if gain == 0:
        return Leaf(dt)

    # If we reach here, we have found a useful feature to partition on
    true_rows, false_rows = partition(dt, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature to ask at this point and the branches that follow depending on the answer
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at nodes
    print(spacing + str(node.question))

    # true branches
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    # false branches
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


my_tree = build_tree(datatable)
print_tree(my_tree)


def classify(row, node):
    # Base case: reaching a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or false-branch
    # Compare feature stored in node to example
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    prob = {}
    for label in counts.keys():
        prob[label] = str(int(counts[label] / total * 100)) + "%"
    return prob


def accuracy(test_data):
    correct = 0  # count # of correct predictions
    for row in test_data:
        #         print ("Actual: %s. Predicted: %s" %
        #        (row[-1], print_leaf(classify(row, my_tree))))
        predict = print_leaf(classify(row, my_tree))
        if row[-1] == list(predict.keys())[0]:
            correct += 1
    accuracy = (correct / len(test_data)) * 100
    print("Accuracy:", accuracy, "%")


accuracy(testing_datatable)
