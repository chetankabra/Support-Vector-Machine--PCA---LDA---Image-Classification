# Chetan Kabra
# Student ID: 1001152872
from __future__ import division
import numpy as np
import os
path_to_dataset = "att_faces/"
#path_to_dataset = "small_data_set/"
import numpy as np
import scipy.misc
import cvxopt
from sklearn.utils import shuffle
import  scipy as sp
import matplotlib.pyplot as plt

class SVM:

    def __init__(self):
        self.all_w = [];
        self.all_b =[];
        self.C = float(100)

    def fit(self, training_data, training_labels):
        n_samples, n_features= training_data.shape
        training_labels = training_labels.reshape(training_labels.shape[1])
        for each in set(training_labels):
            self.modify_training_labels(training_labels, each)
            X = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    X[i, j] = np.dot(training_data[i],training_data[j])
            P = cvxopt.matrix(np.outer(self.new_training_lables, self.new_training_lables.T) * X)
            q = cvxopt.matrix(np.ones(n_samples) * -1)
            A = cvxopt.matrix(self.new_training_lables, (1, n_samples))
            b = cvxopt.matrix(0.0)
            G_std = np.diag(np.ones(n_samples) * -1)
            G_slack = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h_std = np.zeros(n_samples)
            h_slack = np.ones(n_samples) * self.C# soft margin calculation
            h = cvxopt.matrix(np.hstack((h_std, h_slack)))
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            temp_alpha = np.array([])
            for e in solution['x'] :
                temp_alpha = np.hstack([temp_alpha,e])
            temp_alpha.reshape(len(solution['x']),1)
            support_vector_index = temp_alpha > 0.00
            self.alpha = temp_alpha[support_vector_index]
            self.support_vector_labels =  self.new_training_lables[support_vector_index]
            self.support_vector = training_data[support_vector_index]

            # Weight vector
            self.w = np.zeros(n_features)
            for n in range(len(self.alpha)):
                self.w += self.alpha[n] * self.support_vector_labels[n] * self.support_vector[n].astype('float64')
            self.all_w.append(self.w)
            #print (self.all_w.__len__())

            # calculating B value
            new_w = self.w.reshape(1,len(self.w))
            self.b = 0.0
            for index,yi in  enumerate(self.support_vector_labels):
                if yi > -1:
                    self.b = (1 - (np.dot(new_w, self.support_vector[index])))
                    break
            self.all_b.append(self.b)




    def modify_training_labels(self, training_labels, counter):
        self.old_training_labels = np.array(training_labels)
        self.old_training_labels[self.old_training_labels != counter] = -1
        self.old_training_labels[self.old_training_labels == counter] = 1
        self.new_training_lables = self.old_training_labels.astype('double')

    def predict(self, testing_data):
        #n_samples, n_features = testing_data.shape
        output = []
        for i,each in enumerate(testing_data):
            calculate = []
            for w in xrange(0,len(self.all_w)):
                current_prediction = np.dot(each,self.all_w[w]) + self.all_b[w]
                calculate.append(current_prediction)
            max_index = calculate.index(max(calculate))
            output.append(max_index+1)
        return output

    def accuracy(self, predict_labels, testing_labels):
        self.right_predication =0
        for i in range(len(predict_labels)):
            if (predict_labels[i] == testing_labels[i]):
                self.right_predication = self.right_predication + 1
        return (self.right_predication / len(predict_labels) *100)

    def create_training_labels(self,training_labels, counter):
        self.old_training_labels = training_labels
        end  = counter * 5
        start = end-5
        self.new_training_lables = np.ones(self.old_training_labels.__len__()) * -1
        self.new_training_lables[start:end]  = self.new_training_lables[start:end] * -1



class cross_validation:

    def __init__(self,k):
        self.new_training_set = np.array([]);
        self.new_testing_set = np.array([]);
        self.new_training_labels = []
        self.new_testing_labels =[]
        self.K = k;

    def cross_validation_k(self, all_data_set, all_data_labels ):
        self.total_number = all_data_set.shape[0];
        self.partion_ratio = int (self.total_number / self.K) ;
        self.start =0;
        self.end = self.partion_ratio;
        self.accuracy = [];
        # k -fold validation :

        for i in range(0,self.K):
            self.new_testing_set= np.copy(all_data_set.T[:,self.start:self.end])
            self.new_testing_labels = np.copy(all_data_labels.T[:,self.start:self.end])
            range_i = [ x for x in range(self.start,self.end)];
            self.new_training_set =  np.delete(all_data_set.T,range_i, axis=1 )
            self.new_training_labels = np.delete(all_data_labels.T,range_i, axis=1 )
            self.start = self.end
            self.end = self.partion_ratio+self.end

            # PCA on Training set and Testing Set
            self.pca(320, self.new_training_set)

            # LDA
            self.new_lda(self.new_training_set, self.new_training_labels.reshape(self.new_training_labels.shape[1]))

            # KNN
            self.KNN()

            # SVM with linear Kernal
            self.SVM = SVM()
            self.SVM.fit(self.new_training_set.T, self.new_training_labels)
            predicted_labels = self.SVM.predict(self.new_testing_set.T)
            temp = self.SVM.accuracy(predicted_labels, self.new_testing_labels.reshape(self.new_testing_labels.shape[1]))
            self.accuracy.append(temp)

    def pca(self, pca_value, current_data):
        mean_vector = np.mean(current_data, axis=1 )
        cov_matrix = np.cov((current_data.T- mean_vector).T)
        eign_values, eign_vectors = sp.sparse.linalg.eigsh(cov_matrix, k=pca_value)
        # plotting eign_values
        #plt.plot(eign_values)
        #plt.show()
        self.new_training_set = np.dot(eign_vectors.T ,self.new_training_set)
        #self.new_testing_set = (self.new_testing_set.T - mean_vector).T
        self.new_testing_set = np.dot(eign_vectors.T ,self.new_testing_set)

    def eucleadian_distance(self, img1, img2):
        diff = img1-img2
        return np.sqrt(sum(diff ** 2))

    def KNN(self):
        #print self.new_testing_set.shape
        #print self.new_training_set.shape
        accuracy = 0
        for test_image in range(0,self.new_testing_set.shape[1]):
            currect_eucludian = [];
            for i in range(0, self.new_training_set.shape[1]):
                currect_eucludian.append(self.eucleadian_distance(self.new_testing_set.T[test_image], self.new_training_set.T[i]))
            min_distance_index = np.argsort(currect_eucludian)
            #print self.new_testing_labels.shape
            current_predicion = self.new_training_labels[:,min_distance_index[0]]
            #print self.new_testing_labels[:,test_image],current_predicion
            if (current_predicion == self.new_testing_labels[:,test_image]):
                    accuracy +=1
        print accuracy
        self.accuracy.append((accuracy/self.new_testing_set.shape[1])*100)

    def new_lda(self, current_data, current_labels):
        overall_mean = np.mean(current_data, axis=1)
        S_B = self.scatter_between(current_data.T, current_labels)
        S_W = self.scatter_within(current_data.T, current_labels)
        temp1 =np.dot(np.linalg.inv(S_W), S_B)
        eign_values, new_eign_vector = sp.sparse.linalg.eigsh(temp1, k=39)
        self.new_training_set = np.dot(new_eign_vector.T, current_data)
        self.new_testing_set = np.dot(new_eign_vector.T, (self.new_testing_set))


    def comp_mean_vectors(self,X, y):
        class_labels = np.unique(y)
        n_classes = class_labels.shape[0]
        mean_vectors = []
        for cl in class_labels:
            mean_vectors.append(np.mean(X[y == cl], axis=0))
        return mean_vectors

    def scatter_within(self,X, y):
        class_labels = np.unique(y)
        n_features = X.shape[1]
        mean_vectors = self.comp_mean_vectors(X, y)
        S_W = np.zeros((n_features, n_features))
        for cl, mv in zip(class_labels, mean_vectors):
            class_sc_mat = np.zeros((n_features, n_features))
            for row in X[y == cl]:
                row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
                class_sc_mat = class_sc_mat + (row - mv).dot((row - mv).T)
            S_W = S_W + class_sc_mat
        return S_W

    def scatter_between(self,X, y):
        overall_mean = np.mean(X, axis=0)
        n_features = X.shape[1]
        mean_vectors = self.comp_mean_vectors(X, y)
        S_B = np.zeros((n_features, n_features))
        for i, mean_vec in enumerate(mean_vectors):
            list_of_number= []
            for x, val in enumerate(y):
                if val == i+1:
                    list_of_number.append(x)
            n = len(list_of_number)
            mean_vec = mean_vec.reshape(n_features, 1)
            overall_mean = overall_mean.reshape(n_features, 1)
            S_B = S_B +  n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        return S_B


class data_set:
    def __init__(self, path_to_dataset):
        self.training_data = np.array([])
        self.testing_data = np.array([])
        self.training_labels = []
        self.testing_labels =[]
        self.class_type = 0;
        for dir in os.listdir(path_to_dataset):
            self.class_type += 1
            for files in os.listdir(path_to_dataset + "/" + dir):
                check_flag = str(files).split(".")[0]
                if (int(check_flag) <= 6):
                    new_samples = self.read_one_image_and_convert_to_vector(path_to_dataset + "/" + dir + "/" + files)
                    #new_samples = sp.misc.imresize(new_samples, size=2576)
                    if self.training_data.__len__() == 0:
                        self.training_data = new_samples;
                    else:
                        self.training_data = np.hstack([self.training_data, new_samples]);
                    self.training_labels.append(self.class_type)
                else:
                    new_samples = self.read_one_image_and_convert_to_vector(path_to_dataset + "/" + dir + "/" + files)
                    if self.testing_data.__len__() == 0:
                        self.testing_data = new_samples;
                    else:
                        self.testing_data = np.hstack([self.testing_data, new_samples]);
                    self.testing_labels.append(self.class_type)

        self.training_data, self.training_labels  = shuffle(self.training_data.T, self.training_labels)
        self.testing_data, self.testing_labels = shuffle(self.testing_data.T, self.testing_labels)


    def read_one_image_and_convert_to_vector(self,file_name):
        img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
        # resize the image
        #img = scipy.misc.imresize(img, size=[56,46])
        return img.reshape(-1, 1)/255.0 # reshape to column vector and return it




if __name__ == "__main__":
    data_set = data_set(path_to_dataset);
    cross_validation = cross_validation(5);
    all_data_set = np.vstack([data_set.training_data,data_set.testing_data])
    all_labels = np.hstack([data_set.training_labels,data_set.testing_labels])
    all_labels = all_labels.reshape(len(all_labels),1)
    print all_labels.shape
    cross_validation.cross_validation_k(all_data_set,all_labels)
    for i in cross_validation.accuracy:
        print i