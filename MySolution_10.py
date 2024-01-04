import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
### TODO: import any other packages you need for your solution
import cvxpy as cp
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpConstraint, LpConstraintGE, LpConstraintLE, LpConstraintEQ
from pulp import GLPK

#--- Task 1 ---#

# functions to create variables
def create_z_var(i,j):
  return LpVariable(f"z{i}_{j}", lowBound = 0, cat = 'Continuous')


def create_a_var(i,j):
  return LpVariable(f"a{i}_{j}", cat = 'Continuous')


def create_b_var(j):
  return LpVariable(f"b{j}", cat = 'Continuous')

# delta_pj utility function for constraints
def delta(p, j):
    if p > j:
        return 1
    else:
        return 0


class MyClassifier:
    def __init__(self, K):
        self.K = K  # number of classes
        self.n = None # number of inputs (Xs)
        self.m = None # input vector X belongs to R^n
        self.labelY = None
        self.ms = None

        # hyperplanes wTx + b
        self.w = None
        self.b = None

    def arrange(self, trainX, trainY):
        # self.n = trainX.shape[1]
        # self.m = trainX.shape[0]
        self.labelY = []
        self.ms = []

        # Sort trainY and arrange trainX accordingly
        arrangeIndices = np.argsort(trainY)
        arrangedY = np.sort(trainY)
        arrangedX = trainX[arrangeIndices]

        # print(arrangedY[0])
        # print(arrangedX[0])

        # print(self.m)

        for i in range(self.m):
            if len(self.labelY) == 0:
                self.labelY.append(arrangedY[i])
                self.ms.append(1)
            else:
                m_ind = np.argwhere(np.array(self.labelY) == arrangedY[i])
                if len(m_ind) != 0:
                    self.ms[m_ind[0][0]] = self.ms[m_ind[0][0]] + 1
                else:
                    self.ms.append(1)
                    self.labelY.append(arrangedY[i])

        return arrangedX, arrangedY


    def train(self, trainX, trainY):
        self.n = trainX.shape[1]
        self.m = trainX.shape[0]

        # arrange (trainX, trainY) pairs according to classes (m0, m1, m2, m3, mk-1)
        arrangedX, arrangedY = self.arrange(trainX,trainY)

        # create LP model
        model = LpProblem(name="LP", sense=LpMinimize)

        # create LP variables
        z = [[create_z_var(j,i) for i in range(0,self.m)] for j in range(0, self.K-1)]
        # print(z)
        aj_s = [[create_a_var(j,i) for i in range(0,self.n)] for j in range(0, self.K-1)]
        # print(aj_s)
        bj_s = [create_b_var(j) for j in range(0, self.K-1)]
        # print(bj_s)

        # objective function
        obj_fun = lpSum(z[j][i] for i in range(0,self.m) for j in range(0, self.K-1))
        # print(obj_fun)
        model += obj_fun
        # print(model)

        # Constraints
        eps = 1
        m_sum = 0

        for p in range(self.K):
            for i in range(m_sum, m_sum + self.ms[p]):
                for j in range(self.K - 1):
                    model += LpConstraint( z[j][i] + np.power((-1),delta(p,j)) * (np.dot(aj_s[j], arrangedX[i,:]) + bj_s[j]), sense = LpConstraintGE, rhs = eps )
            m_sum += self.ms[p]

        # print(model)

        # Solve the problem
        # status = model.solve(solver=GLPK(msg=False))
        status = model.solve()
        # print(f"status: {model.status}, {LpStatus[model.status]}")
        # print(f"objective: {model.objective.value()}")
        # for var in model.variables():
        #     print(f"{var.name}: {var.value()}")

        # set the trained (optimized) hyperplanes
        vars = model.variables()
        self.w = np.zeros((self.K-1,self.n))
        self.b = np.zeros((self.K-1))
        for i in range (self.K-1):
          self.w[i][0] = vars[2*i].value()
          self.w[i][1] = vars[2*i+1].value()
          self.b[i] = vars[2*(self.K-1) + i].value()

    def predict(self, testX):
        for i in range(self.K-1):
            if np.dot(self.w[i], testX) + self.b[i] >= 0:
              predY = self.labelY[i]
              return predY
        predY = self.labelY[-1]
        return predY


    def evaluate(self, testX, testY):
        predYs = np.zeros([len(testX),1])
        for i in range(len(testX)):
            predYs[i] = self.predict(testX[i])
        accuracy = accuracy_score(testY, predYs)
        return accuracy


class MyClassifier_v2:
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.w = None
        self.b = None
        self.M = 1e6
        self.labels = None

    def train(self, trainX, trainY):
        ''' Task 1-2
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm
        '''
        n = trainX.shape[0]  # number of data points
        m = trainX.shape[1]  # size of each data point

        # pre processing
        self.labels = np.unique(trainY)
        Y = np.zeros((n, self.K))
        for index, label in np.ndenumerate(self.labels):
            Y[trainY == label, index] = 1

        # Variable
        T = cp.Variable((n, self.K))
        Z = cp.Variable((n, self.K))
        W = cp.Variable((self.K, m))
        B = cp.Variable(self.K)

        # Objective function
        obj = cp.Minimize(cp.sum(T) + cp.sum(Z))

        # Constraints
        constraints = [
            T >= 0,
            T >= 1 - self.M * (1 - Y) - (trainX @ W.T + B[None, :]),
            Z >= 0,
            Z >= 1 - self.M * Y + (trainX @ W.T + B[None, :]),

        ]

        # Solve
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        self.w = W.value
        self.b = B.value

    def predict(self, testX):
        ''' Task 1-2
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        predY = np.argmax(np.dot(testX, self.w.transpose()) + self.b, axis=1)
        predY = self.labels[predY]

        # Return the predicted class labels of the input data (testX)
        return predY

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        self.cluster_centers_ = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''

        # STEP1 - initialization: randomly select K points from trainX (initial centroids)
        n = trainX.shape[0]  # number of data points
        m = trainX.shape[1]  # size of each data point
        self.cluster_centers_ = trainX[np.random.choice(n, size=self.K, replace=False)]

        # STEP2 - iteration of assignment and update
        itr = 0
        max_itr = 1e6
        trh = 1e-6
        while itr < max_itr:
            # iteration
            itr = itr + 1

            # distance d_ij
            vector_from_centers = trainX.reshape(n, 1, m) - self.cluster_centers_  # n x K x m
            distance_from_centers = np.linalg.norm(vector_from_centers, axis=2)  # n x K
            d_ij = distance_from_centers.flatten()

            # STEP2.1 - Assignment (Linear Programming)
            # constraint 1: each data has to be assigned to one cluster
            A_1 = np.zeros((n, n * self.K))
            for i in range(n):
                A_1[i, i * self.K: (i + 1) * self.K] = np.ones(self.K)
            b_1 = np.ones(n)
            # constraint 2: a cluster includes at least one data
            A_2 = np.tile(np.eye(self.K), n)
            b_2 = np.ones(self.K)
            # constraint 3: integer
            A_3 = np.eye(n * self.K)
            b_3 = np.zeros(n * self.K)
            A_4 = np.eye(n * self.K)
            b_4 = np.ones(n * self.K)
            # variable x_ij
            x_ij = cp.Variable(n * self.K)
            # optimization problem
            prob = cp.Problem(cp.Minimize(d_ij.T @ x_ij),
                              [A_1 @ x_ij == b_1,
                               A_2 @ x_ij >= b_2,
                               A_3 @ x_ij >= b_3,
                               A_4 @ x_ij <= b_4])
            prob.solve(solver=cp.ECOS)
            # assign labels
            x_ij = x_ij.value.reshape((n, self.K))
            new_labels = np.argmax(x_ij, axis=1)

            # update cluster centroids
            new_cluster_centers_ = np.zeros_like(self.cluster_centers_)
            for j in np.arange(self.K):
                new_cluster_centers_[j, :] = np.mean(trainX[new_labels == j], axis=0)

            # convergence criteria
            if np.max(np.linalg.norm(self.cluster_centers_ - new_cluster_centers_, axis=1)) < trh:
                self.cluster_centers_ = new_cluster_centers_
                self.labels = new_labels
                break
            else:
                self.cluster_centers_ = new_cluster_centers_
                self.labels = new_labels
            # if np.all(new_labels == self.labels):
            #     break
            # else:
            #     self.labels = new_labels

        # Update and return the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # distance of points from centroids n x K
        vector_from_centers = testX.reshape(testX.shape[0], 1, testX.shape[1]) - self.cluster_centers_  # n x K x m
        distance_from_centers = np.linalg.norm(vector_from_centers, axis=2)  # n x K
        pred_labels = np.argmin(distance_from_centers, axis=1)

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels.astype(int)[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables


##########################################################################
# --- Task 3 ---#
class MyLabelSelection:
    def __init__(self, selectRatio, centroids):
        self.ratio = selectRatio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

        # Distance matrix
        self.D = None  # Dimensions: num_classes x N
        self.centroids = centroids

    def select(self, trainX):
        ''' Task 3-2'''
        # Num of data points and clusters
        N = trainX.shape[0]
        K = self.centroids.shape[0]

        # Parameters
        self.D = self.find_distance_to_centroid(trainX, self.centroids)
        self.C = self.min_distance_matrix(self.D)

        # Variable
        x = cp.Variable((K, N))

        # Objective function
        obj = cp.Maximize(cp.trace(self.C @ x.T))

        # Constraints
        constraints = [
            x >= 0,
            x <= 1,
            cp.sum(x, axis=0) <= 1,
            cp.sum(x, axis=1) <= np.round(N * self.ratio / K),
            cp.sum(x, axis=1) >= np.round(N * self.ratio / K)
        ]

        # Solve
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        x_numpy = np.sum(x.value, axis=0)

        # Return an index list that specifies which data points to label
        data_to_label = np.where(x_numpy.round() == 1)[0]
        data_to_label = list(data_to_label)

        return data_to_label

    def find_distance_to_centroid(self, X_train, centroids):
        X_train = np.array(X_train)

        # number of samples
        N = X_train.shape[0]

        # number of clusters
        K = centroids.shape[0]

        # Initialize matrix
        D = np.zeros((K, N))

        # Calculate dissimilarity using L2 norm (Euclidean distance)
        for i in range(K):
            for j in range(N):
                D[i, j] = np.linalg.norm(centroids[i] - X_train[j])

        return D
    
    def min_distance_matrix(self, D):
        min_values = np.min(D, axis=0)

        # Preserve min values
        D_modified = np.where(D > min_values, 0, D)

        return D_modified


    