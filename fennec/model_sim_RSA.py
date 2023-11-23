# using RSA to calculate model similarity

import os
import pickle
from joblib.logger import PrintTime
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy import linalg
from tqdm import tqdm

import sklearn
from sklearn import preprocessing
import sklearn.preprocessing
import sklearn.decomposition

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import NeighborhoodComponentsAnalysis,KNeighborsClassifier
import utils
import constants



import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

# sys.path.insert(0, dir_path)
sys.path.insert(0, parent_dir_path)
print(sys.path)
import methods


class Model_Sim_RSA:
    def __init__(self, pre_path='./cache/probes/fixed_budget_500'):
        print('calculate model similarity using RSA based on forward features')
        self.pre_path = pre_path
        self.architecture = constants.variables['Architecture']
        self.source_datasets = constants.variables['Source Dataset']
        self.target_datasets = constants.variables['Target Dataset']
        self.models_name = [
            a + '%' + b for a in self.architecture for b in self.source_datasets]
        
    def feature_reduce(self, features: np.ndarray, f: int = None) -> np.ndarray:
        """
        Use PCA to reduce the dimensionality of the features.

        If f is none, return the original features.
        If f < features.shape[0], default f to be the shape.
        """
        if f is None:
            return features

        if f > features.shape[0]:
            f = features.shape[0]

        return sklearn.decomposition.PCA(
            n_components=f,
            svd_solver='randomized',
            random_state=1919,
            iterated_power=1).fit_transform(features)

    def get_features(self, target_dataset, run=0):
        # feature file: alexnet_caltech101_caltech101_0.pkl  architecture, source, target, run.pkl
        # 32  'stanford_dogs%googlenet', 'stanford_dogs%alexnet', 'voc2007%resnet50'...
  
        res = {}
        for name in self.models_name:
            archi, source = name.split('%')
            feature_name = f'{archi}_{source}_{target_dataset}_{run}.pkl'
            with open(os.path.join(self.pre_path, feature_name), 'rb') as f:
                fe = pickle.load(f)
                res[name] = fe['features']
                #__import__('IPython').embed()
        return res

    def calculate_RSA(self, feats1, feats2, pca=0):
        if pca != 0:
            feats1 = self.feature_reduce(feats1, pca)
            feats2 = self.feature_reduce(feats2, pca)

        scaler = sklearn.preprocessing.StandardScaler()
        feats1 = scaler.fit_transform(feats1)
        feats2 = scaler.fit_transform(feats2)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(feats2)
        #print(rdm1.shape)
        def get_lowertri(rdm):
            num_conditions = rdm.shape[0]
            return rdm[np.triu_indices(num_conditions, 1)]

        lt_rdm1 = get_lowertri(rdm1)
        lt_rdm2 = get_lowertri(rdm2)

        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0]

    def cal_model_similarity(self, target, pca_dim):
        print(f'RSA calculating {target}, dimension deduction {pca_dim}')
        model_size = len(self.models_name)
        similarity_matrix = np.zeros((model_size, model_size), dtype=np.float)
        features = self.get_features(target)
        for i in range(model_size):
            for j in range(model_size):
                if self.models_name[i] == self.models_name[j]:
                    similarity_matrix[i][j] = 1
                else:
                    f1 = features[self.models_name[i]]
                    f2 = features[self.models_name[j]]
                    score = self.calculate_RSA(f1, f2, pca_dim)
                    
                    similarity_matrix[i][j] = score
        return similarity_matrix


from sklearn.cluster import KMeans
from sklearn import metrics
class Cluster(Model_Sim_RSA):
    def __init__(self, pre_path):
        print("check the cluster performance as matrix completion")
        # https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
        super(Cluster, self).__init__(pre_path=pre_path)
        

    def get_features_labels(self, target_dataset, run=0):
        # get training set and
        
        features = {}
        true_labels = {}
        for name in self.models_name:
            archi, source = name.split('%')
            feature_name = f'{archi}_{source}_{target_dataset}_{run}.pkl'
            with open(os.path.join(self.pre_path, feature_name), 'rb') as f:
                fe = pickle.load(f)
                features[name] = fe['features']
                true_labels[name] = fe['y']
        return features, true_labels

    def without_kmeans(self, features, true_labels, pca_dim=32):

        model_size = len(self.models_name)
        res = np.zeros((model_size, ))
        # calculate the column performance
        for idx, model_name in enumerate(self.models_name):
            feat = features[model_name]
            label = true_labels[model_name]
            #assert feat.shape[0] == 500, 'wrong features!'
            if pca_dim > 0:
                feat = self.feature_reduce(feat, pca_dim)
            assert feat.shape[1] == pca_dim
            res[idx] = metrics.silhouette_score(feat, label, metric='euclidean') 
        return res

    def with_kmeans(self, features, true_labels, pca_dim=32):
        model_size = len(self.models_name)
        res = np.zeros((model_size, ))
        for idx, model_name in enumerate(self.models_name):
            feat = features[model_name]
            true_label = true_labels[model_name]
            if pca_dim > 0:
                feat = self.feature_reduce(feat, pca_dim)
            assert feat.shape[1] == pca_dim
            # do the kmeans:
            kmeans_model = KMeans(n_clusters=len(list(set(true_label))), random_state=1).fit(feat)
            pred_label = kmeans_model.labels_
            res[idx] = metrics.adjusted_mutual_info_score(true_label, pred_label) 

            
        return res

    def cal_cluster_performance(self, target):
        model_cluster_peformance = np.zeros((len(self.models_name), len(self.target_datasets)))
        for idx, target_dataset in enumerate(self.target_datasets):
            # get all forward features related on target dataset
            features, labels = self.get_features_labels(target_dataset=target_dataset)
            # calculate cluster score
            scores = self.without_kmeans(features=features, true_labels=labels)
            model_cluster_peformance[:, idx] = scores

        return model_cluster_peformance
        
    
    def completion(self):
        pass


def feature_reduce(features:np.ndarray, f:int=None) -> np.ndarray:
    """
    Use PCA to reduce the dimensionality of the features.

    If f is none, return the original features.
    If f < features.shape[0], default f to be the shape.
    """
    if f is None:
        return features

    if f > features.shape[0]:
        f = features.shape[0]

    return sklearn.decomposition.PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)



class FDA_model():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components


    def _cov(self, X, shrinkage=-1):
        emp_cov = np.cov(np.asarray(X).T, bias=1)
        if shrinkage < 0:
            return emp_cov
        n_features = emp_cov.shape[0]
        mu = np.trace(emp_cov) / n_features # https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/covariance/_shrunk_covariance.py#L82
        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
        return shrunk_cov

    def iterative_A(self, A, max_iterations=3):
        '''
        calculate the largest eigenvalue of A
        '''
        x = A.sum(axis=1)
        #k = 3
        for _ in range(max_iterations):
            temp = np.dot(A, x)
            y = temp / np.linalg.norm(temp, 2)
            temp = np.dot(A, y)
            x = temp / np.linalg.norm(temp, 2)
        return np.dot(np.dot(x.T, A), y)

    def softmax(self, X, copy=True):
        if copy:
            X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X


    def _class_means(self, X, y):
        """Compute class means.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        Returns
        -------
        means : array-like of shape (n_classes, n_features)
            Class means.
        means ： array-like of shape (n_classes, n_features)
            Outer classes means.
        """
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]

        means_ = np.zeros(shape=(len(classes), X.shape[1]))
        for i in range(len(classes)):
            means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)    
        return means, means_
     
    def _solve_eigen(self, X, y, shrinkage):

        classes, y = np.unique(y, return_inverse=True) # classes==len(label)    y= map using classes,len==X.shape[0]
        cnt = np.bincount(y)
        
        means = np.zeros(shape=(len(classes), X.shape[1]))  # (10, 32)
        np.add.at(means, y, X) 
        means /= cnt[:, None]  # calcuate the mean of each class.  cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(self._cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = self.iterative_A(Sw, max_iterations=3) 
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)  # a=5
            self.shrinkage = shrinkage    # lambda λ in paper
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        #print("Shrinkage: {}".format(shrinkage))
        # between scatter
        self.shrinkage = 0   # no reg!!!
        St = self._cov(X, shrinkage=self.shrinkage)   
        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter
        
        evals, evecs = linalg.eigh(Sb, shrunk_Sw)

        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors disasc
    
        self.scalings_ = evecs # projection matrix
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N

        '''
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])  

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)
 
        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return self.softmax(scores)


class FDA():
    def __init__(self, n_dims=None) -> None:
        # code from https://github.com/TencentARC/SFDA
        # FDA : no lambda 
        #       no confmix
        self.n_dims = n_dims

    def forward_sklearn(self, features, y):
        self.features = feature_reduce(features, self.n_dims)
        X = self.features
        # to enable in-consistent label space, like 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        n = len(y)
        num_classes = np.unique(y)

        clf = LinearDiscriminantAnalysis(solver='eigen')    
        clf.fit(X, y)

        score = clf.score(X, y)
        return score


    def forward(self, features, y) -> float:
        self.features = feature_reduce(features, self.n_dims)
        X = self.features
        # to enable in-consistent label space, like 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        n = len(y)
        num_classes = len(np.unique(y))
        
        FDA_first = FDA_model()
        prob = FDA_first.fit(X, y).predict_proba(X)  # p(y|x)


        fda_score = np.sum(prob[np.arange(n), y]) / n 
        return fda_score


class FDA_score(Cluster):
    def __init__(self, pre_path='./cache/probes/fixed_budget_500'):
        print(f"cal FDA using {pre_path}")
        self.fda_matrix = None
        super(FDA_score, self).__init__(pre_path=pre_path)
    
    def cal_FDA(self, pca_dim, run):
        # 1. get forward features
        self.fda_matrix = np.zeros((len(self.models_name), len(self.target_datasets)), dtype=np.float)

        for idx, data in enumerate(self.target_datasets):
            features, labels = self.get_features_labels(data, run=run)  # [model_sizes.]
            for j, model_name in enumerate(features):
                # print(f'doing {data} with {model_name} ')
                model = FDA(n_dims=pca_dim)
                self.fda_matrix[j][idx] = model.forward(features[model_name], labels[model_name])
        return self.fda_matrix

