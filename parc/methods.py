import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

import constants
import datasets

import scipy.stats
from scipy import linalg
import sklearn.neighbors
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils.extmath import svd_flip
# code url at https://github.com/thuml/LogME/blob/main/LogME.py
import warnings
from numba import njit
# This is for Logistic so it doesn't complain that it didn't converge
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


def split_data(data:np.ndarray, percent_train:float):
    split = data.shape[0] - int(percent_train * data.shape[0])
    return data[:split], data[split:]

class TransferabilityMethod:	
    def __call__(self, 
        features:np.ndarray, probs:np.ndarray, y:np.ndarray,
        source_dataset:str, target_dataset:str, architecture:str,
        cache_path_fn) -> float:
        
        self.features = features
        self.probs = probs
        self.y = y

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.architecture = architecture

        self.cache_path_fn = cache_path_fn

        # self.features = sklearn.preprocessing.StandardScaler().fit_transform(self.features)
        le = preprocessing.LabelEncoder()
        self.y = le.fit_transform(self.y)
        return self.forward()

    def forward(self) -> float:
        raise NotImplementedError




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




class LEEP(TransferabilityMethod):
    """
    LEEP: https://arxiv.org/abs/2002.12462
    src ('probs', 'features') denotes what to use for leep.
    normalization ('l1', 'softmax'). The normalization strategy to get everything to sum to 1.
    """

    def __init__(self, n_dims:int=None, src:str='probs', normalization:str=None, use_sigmoid:bool=False):
        self.n_dims = n_dims
        self.src = src
        self.normalization = normalization
        self.use_sigmoid = use_sigmoid

    def forward(self) -> float:
        
        theta = getattr(self, self.src)
        y = self.y
        
        n = theta.shape[0]
        n_y = constants.num_classes[self.target_dataset]

        # n             : Number of target data images
        # n_z           : Number of source classes
        # n_y           : Number of target classes
        # theta [n, n_z]: The source task probabilities on the target images
        # y     [n]     : The target dataset label indices {0, ..., n_y-1} for each target image

        unnorm_prob_joint    = np.eye(n_y)[y, :].T @ theta                       # P(y, z): [n_y, n_z]
        unnorm_prob_marginal = theta.sum(axis=0)                                 # P(z)   : [n_z]
        prob_conditional     = unnorm_prob_joint / unnorm_prob_marginal[None, :] # P(y|z) : [n_y, n_z]

        leep = np.log((prob_conditional[y] * theta).sum(axis=-1)).sum() / n      # Eq. 2
        # __import__('IPython').embed()
        return leep


class NLEEP(TransferabilityMethod):
    """
    https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Ranking_Neural_Checkpoints_CVPR_2021_paper.pdf
    """
    def __init__(self, n_dims:int=None, src:str='probs', normalization:str=None, use_sigmoid:bool=False, random_state:int=123):
        self.n_dims = n_dims
        self.src = src
        self.normalization = normalization
        self.use_sigmoid = use_sigmoid
        self.random_state = random_state
        # the number of Gaussian components five times the class number of a downstream task.
        
        
    
    def forward(self) -> float:
        self.num_components_gmm = constants.num_classes[self.target_dataset] * 5 
        self.num_components_gmm = 50
        gmm = GaussianMixture(
                    n_components=self.num_components_gmm, random_state=self.random_state).fit(self.features)
        
        gmm_predictions = gmm.predict_proba(self.features)
        theta = gmm_predictions.astype('float32')
        #__import__('IPython').embed()
        # normal nleep:
        y = self.y
        
        n = theta.shape[0]
        n_y = constants.num_classes[self.target_dataset]

        # n             : Number of target data images
        # n_z           : Number of source classes
        # n_y           : Number of target classes
        # theta [n, n_z]: The source task probabilities on the target images
        # y     [n]     : The target dataset label indices {0, ..., n_y-1} for each target image

        unnorm_prob_joint    = np.eye(n_y)[y, :].T @ theta                       # P(y, z): [n_y, n_z]
        unnorm_prob_marginal = theta.sum(axis=0)                                 # P(z)   : [n_z]
        prob_conditional     = unnorm_prob_joint / unnorm_prob_marginal[None, :] # P(y|z) : [n_y, n_z]

        leep = np.log((prob_conditional[y] * theta).sum(axis=-1)).sum() / n      # Eq. 2
        return leep


class GBC(TransferabilityMethod):
    def __init__(self, gaussian_type=None, n_dims=None):
        assert gaussian_type in ('diagonal', 'spherical')
        self.gaussian_type = gaussian_type
        self.n_dims = n_dims

    def forward(self) -> float:
        self.features = feature_reduce(self.features, self.n_dims)
        unique_labels, _ = tf.unique(self.y)
        unique_labels = list(unique_labels)
        per_class_stats = self.compute_per_class_mean_and_variance(
            self.features, self.y, unique_labels
        )

        per_class_bhattacharyya_distance = []
        for c1 in unique_labels:
            tmp_metric = []
            for c2 in unique_labels:
                if c1 != c2:
                    bhattacharyya_distance = self.get_bhattacharyya_distance(
                        per_class_stats, int(c1), int(c2), self.gaussian_type
                    )
                    tmp_metric.append(tf.exp(-bhattacharyya_distance))
            per_class_bhattacharyya_distance.append(tf.reduce_sum(tmp_metric))
        
        gbc = - tf.reduce_sum(per_class_bhattacharyya_distance)
        #__import__('IPython').embed()
        return gbc.numpy()


    def compute_per_class_mean_and_variance(self, features, target_labels, unique_labels):
        per_class_stats = {}
        for label in unique_labels:
            label = int(label)
            per_class_stats[label] = {}
            class_ids = tf.equal(target_labels, label)
            class_features = tf.gather_nd(features, tf.where(class_ids))
            mean = tf.reduce_mean(class_features, axis=0)
            variance = tf.math.reduce_variance(class_features, axis=0)
            per_class_stats[label]['mean'] = mean
            per_class_stats[label]['variance'] = tf.maximum(variance, 1e-4)
        return per_class_stats

    def get_bhattacharyya_distance(self, per_class_stats, c1, c2, gaussian_type):
        mu1 = per_class_stats[c1]['mean']
        mu2 = per_class_stats[c2]['mean']
        sigma1 = per_class_stats[c1]['variance']
        sigma2 = per_class_stats[c2]['variance']
        if gaussian_type == 'spherical':
            sigma1 = tf.reduce_mean(sigma1)
            sigma2 = tf.reduce_mean(sigma2)
        return self.compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2)

    def compute_bhattacharyya_distance(self, mu1, mu2, sigma1, sigma2):
        avg_sigma = (sigma1 + sigma2) / 2
        first_part = tf.reduce_sum((mu1 - mu2)**2 / avg_sigma) / 8
        second_part = tf.reduce_sum(tf.math.log(avg_sigma))
        second_part -= 0.5 * (tf.reduce_sum(tf.math.log(sigma1)))
        second_part -= 0.5 * (tf.reduce_sum(tf.math.log(sigma2)))
        return first_part + 0.5 * second_part


class NegativeCrossEntropy(TransferabilityMethod):
    """ NCE: https://arxiv.org/pdf/1908.08142.pdf """

    def forward(self, eps=1e-5) -> float:
        z = self.probs.argmax(axis=-1)

        n = self.y.shape[0]
        n_y = constants.num_classes[self.target_dataset]
        n_z = constants.num_classes[self.source_dataset]

        prob_joint    = (np.eye(n_y)[self.y, :].T @ np.eye(n_z)[z, :]) / n + eps  # [10,555]
        prob_marginal = np.eye(n_z)[z, :].sum(axis=0) / n + eps  # [555,]

        NCE = (prob_joint * np.log(prob_joint / prob_marginal[None, :])).sum()
        

        return NCE


class TransRate(TransferabilityMethod):
    def __init__(self, n_dims:int=None):
        self.n_dims = n_dims

    def coding_rate(self, Z, eps=1e-4):
        n, d = Z.shape
        (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
        return 0.5 * rate

    def forward(self, eps=1e-3):
        self.features = feature_reduce(self.features, self.n_dims)

        Z = self.features
        Z = Z - np.mean(Z, axis=0, keepdims=True)
        RZ = self.coding_rate(Z, eps)
        RZY = 0.
        K = int(self.y.max() + 1)

        for i in range(K):
            RZY += self.coding_rate(Z[(self.y == i).flatten()], eps)
        return RZ - RZY / K


class HScore(TransferabilityMethod):
    """ HScore from https://ieeexplore.ieee.org/document/8803726 """

    def __init__(self, n_dims:int=None, use_published_implementation:bool=False):
        self.use_published_implementation = use_published_implementation
        self.n_dims = n_dims

    def getCov(self, X):
        X_mean= X - np.mean(X,axis=0,keepdims=True)
        cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
        return cov

    def getHscore(self, f,Z):
        Covf = self.getCov(f)
        g = np.zeros_like(f)
        for z in range(constants.num_classes[self.target_dataset]):
            idx = (Z == z)
            if idx.any():
                Ef_z=np.mean(f[idx, :], axis=0)
                g[idx]=Ef_z
        
        Covg=self.getCov(g)
        score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg))

        return score

    def get_hscore_fast(self, eps=1e-8):
        # The original implementation of HScore isn't properly vectorized, so do that here
        cov_f = self.getCov(self.features)
        n_y = constants.num_classes[self.target_dataset]

        # Vectorize the inner loop over each class
        one_hot_class = np.eye(n_y)[self.y, :]   # [#probe, #classes]
        class_counts = one_hot_class.sum(axis=0) # [#classes]

        # Compute the mean feature per class
        mean_features = (one_hot_class.T @ self.features) / (class_counts[:, None] + eps) # [#classes, #features]

        # Redistribute that into the original features' locations
        g = one_hot_class @ mean_features # [#probe, #features]
        cov_g = self.getCov(g)
        
        score = np.trace(np.linalg.pinv(cov_f, rcond=1e-15) @ cov_g)

        return score
        

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)

        scaler = sklearn.preprocessing.StandardScaler()
        self.features = scaler.fit_transform(self.features)

        if self.use_published_implementation:
            return self.getHscore(self.features, self.y)
        else:
            return self.get_hscore_fast()



class kNN(TransferabilityMethod):
    """
    k Nearest Neighbors with hold-one-out cross-validation.

    Metric can be one of (euclidean, cosine, cityblock)

    This method supports VOC2007.
    """

    def __init__(self, k:int=1, metric:str='l2', n_dims:int=None):
        self.k = k
        self.metric = metric
        self.n_dims = n_dims
    
    def forward(self) -> float:
        self.features = feature_reduce(self.features, self.n_dims)

        dist = sklearn.metrics.pairwise_distances(self.features, metric=self.metric)
        idx = np.argsort(dist, axis=-1)

        # After sorting, the first index will always be the same element (distance = 0), so choose the k after
        idx = idx[:, 1:self.k+1]

        votes = self.y[idx]
        preds, counts = scipy.stats.mode(votes, axis=1)

        n_data = self.features.shape[0]

        preds = preds.reshape(n_data, -1)
        counts = counts.reshape(n_data, -1)
        votes = votes.reshape(n_data, -1)

        preds = np.where(counts == 1, votes, preds)

        return 100*(preds == self.y.reshape(n_data, -1)).mean()
        # return -np.abs(preds - self.y).sum(axis=-1).mean() # For object detection


class SplitkNN(TransferabilityMethod):
    """ k Nearest Neighbors using a train-val split using sklearn. Only supports l2 distance. """

    def __init__(self, percent_train:float=0.5, k:int=1, n_dims:int=None):
        self.percent_train = percent_train
        self.k = k
        self.n_dims = n_dims

    def forward(self) -> float:
        self.features = feature_reduce(self.features, self.n_dims)

        train_x, test_x = split_data(self.features, self.percent_train)
        train_y, test_y = split_data(self.y       , self.percent_train)

        nn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.k).fit(train_x, train_y)
        return 100*(nn.predict(test_x) == test_y).mean()


class SplitLogistic(TransferabilityMethod):
    """ Logistic classifier using a train-val split using sklearn. """

    def __init__(self, percent_train:float=0.5, n_dims:int=None):
        self.percent_train = percent_train
        self.n_dims = n_dims
        
    def forward(self) -> float:
        self.features = feature_reduce(self.features, self.n_dims)
        
        train_x, test_x = split_data(self.features, self.percent_train)
        train_y, test_y = split_data(self.y       , self.percent_train)

        logistic = sklearn.linear_model.LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', max_iter=20, tol=1e-1).fit(train_x, train_y)
        return 100*(logistic.predict(test_x) == test_y).mean()


class RSA(TransferabilityMethod):
    """
    Computes the RSA similarity metric proposed in https://arxiv.org/abs/1904.11740. 

    Note that this requires the probes to be fully extracted before running.

    This method supports VOC2007.
    """
    def __init__(self, reference_architecture:str=None, n_dims:int=None):
        self.reference_architecture = reference_architecture
        self.n_dims = n_dims

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        
        reference_architecture = self.reference_architecture if self.reference_architecture is not None else self.architecture

        with open(self.cache_path_fn(reference_architecture, self.target_dataset, self.target_dataset), 'rb') as f:
            reference_params = pickle.load(f)
        
        reference_features = reference_params['features']
        reference_features = feature_reduce(reference_features, self.n_dims)
        
        return self.get_rsa_correlation(self.features, reference_features)
    
    def get_rsa_correlation(self, feats1:np.ndarray, feats2:np.ndarray) -> float:
        scaler = sklearn.preprocessing.StandardScaler()
        
        feats1 = scaler.fit_transform(feats1)
        feats2 = scaler.fit_transform(feats2)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(feats2)

        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)

        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100
    
    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions,1)] # return the upper triangular part starting at the main diagonal


class PARC(TransferabilityMethod):
    """
    Computes PARC, a variation of RSA that uses target labels instead of target features to cut down on training time.
    This was presented in this paper.
    
    This method supports VOC2007.
    """

    def __init__(self, n_dims:int=None, fmt:str=''):
        self.n_dims = n_dims
        self.fmt = fmt

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        
        num_classes = constants.num_classes[self.target_dataset]    # [500, 2048]
        labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y # [500, 10]
        return self.get_parc_correlation(self.features, labels)

    def get_parc_correlation(self, feats1, labels2):
        scaler = sklearn.preprocessing.StandardScaler()

        feats1  = scaler.fit_transform(feats1)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(labels2)
        
        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)
        
        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions,1)]


class DDS(TransferabilityMethod):
    """
    DDS from https://github.com/cvai-repo/duality-diagram-similarity/
    
    This method supports VOC2007.
    """

    def __init__(self, reference_architecture:str=None, n_dims:int=None):
        self.reference_architecture = reference_architecture
        self.n_dims = n_dims

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        
        reference_architecture = self.reference_architecture if self.reference_architecture is not None else self.architecture

        with open(self.cache_path_fn(reference_architecture, self.target_dataset, self.target_dataset), 'rb') as f:
            reference_params = pickle.load(f)
        
        reference_features = reference_params['features']
        reference_features = feature_reduce(reference_features, self.n_dims)
        
        return self.get_similarity_from_rdms(self.features, reference_features)

    
    def rdm(self, activations_value,dist):
        """
        Parameters
        ----------
        activations_value : numpy matrix with dimensions n x p 
			task 1 features (n = number of images, p = feature dimensions) 
        dist : string
            distance function to compute dissimilarity matrix
        Returns
        -------
        RDM : numpy matrix with dimensions n x n 
            dissimilarity matrices
        """
        if dist == 'pearson':
            RDM = 1 - np.corrcoef(activations_value)
        elif dist == 'cosine':
            RDM = 1 - sklearn.metrics.pairwise.cosine_similarity(activations_value)
        return RDM


    def get_similarity_from_rdms(self, x,y,debiased=True,centered=True):
        """
        Parameters
        ----------

            task 1 features (n = number of images, p = feature dimensions) 
        y : numpy matrix with dimensions n x p
            task 1 features (n = number of images, p = feature dimensions) 
        dist : string
            distance function to compute dissimilarity matrices
        feature_norm : string
            feature normalization type
        debiased : bool, optional
            set True to perform unbiased centering 
        centered : bool, optional
            set True to perform unbiased centering 
        Returns
        -------
        DDS: float
            DDS between task1 and task2 
        """
        x = sklearn.preprocessing.StandardScaler().fit_transform(x)
        y = sklearn.preprocessing.StandardScaler().fit_transform(y)
        
        return self.cka(self.rdm(x, 'cosine'), self.rdm(y, 'cosine'), debiased=debiased,centered=centered) * 100
        

    def center_gram(self, gram, unbiased=False):
        """
        Center a symmetric Gram matrix.
        
        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.
        
        Args:
            gram: A num_examples x num_examples symmetric matrix.
            unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.
        Returns:
            A symmetric matrix with centered columns and rows.
        
        P.S. Function from Kornblith et al., ICML 2019
        """
        if not np.allclose(gram, gram.T):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.copy()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]
   
        return gram


    def cka(self, gram_x, gram_y, debiased=False,centered=True):
        """
        Compute CKA.
        Args:
            gram_x: A num_examples x num_examples Gram matrix.
            gram_y: A num_examples x num_examples Gram matrix.
            debiased: Use unbiased estimator of HSIC. CKA may still be biased.
        Returns:
            The value of CKA between X and Y.
            P.S. Function from Kornblith et al., ICML 2019
        """
        if centered:
            gram_x = self.center_gram(gram_x, unbiased=debiased)
            gram_y = self.center_gram(gram_y, unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        
        return scaled_hsic / (normalization_x * normalization_y)


    
class LearnedHeuristic():

    def __init__(self, cache_file:str='./cache/learned_heuristic.pkl'):
        self.cache_file = cache_file

    def predict(self, x:list) -> float:
        return sum([a * x_i for a, x_i in zip(self.coeffs, x)])

    def make_feature(self, arch:str, source:str, target:str) -> list:
        feats = [
            constants.num_classes[source],
            constants.num_classes[target],
            constants.dataset_images[source],
            constants.dataset_images[target],
            constants.model_layers[arch]
        ]

        return feats + [np.log(x) for x in feats]

    def fit(self, oracle_path:str, percent_train:float=0.5):
        oracle = pd.read_csv(oracle_path)

        x = []
        y = []

        for idx, row in oracle.iterrows():
            arch   = row['Architecture']
            source = row['Source Dataset']
            target = row['Target Dataset']

            x.append(self.make_feature(arch, source, target))
            y.append(row['Oracle'])
        
        x = np.array(x)
        y = np.array(y)

        regr = sklearn.linear_model.LinearRegression()
        regr.fit(x, y)
        
        self.coeffs = list(regr.coef_)

        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.coeffs, f)
    
    def load(self):
        with open(self.cache_file, 'rb') as f:
            self.coeffs = pickle.load(f)


class SFDA_model():
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
        np.add.at(means, y, X) # https://blog.csdn.net/qq_42698422/article/details/101062718 means += y{index} + X{value}
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
            largest_evals_w = self.iterative_A(Sw, max_iterations=3) # https://blog.csdn.net/seventonight/article/details/116268689
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)  # a=5
            self.shrinkage = shrinkage    # lambda λ in paper
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        #print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = self._cov(X, shrinkage=self.shrinkage)   #https://zhuanlan.zhihu.com/p/35566052

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter
        
        evals, evecs = linalg.eigh(Sb, shrunk_Sw)

        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors disasc
        # print(evecs)
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_)

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


class SFDA(TransferabilityMethod):
    def __init__(self, n_dims=None) -> None:
        # code from https://github.com/TencentARC/SFDA
        self.n_dims = n_dims

    def forward(self) -> float:
        self.features = feature_reduce(self.features, self.n_dims)
        X = self.features
        y = self.y
        # to enable in-consistent label space, like 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        n = len(y)
        num_classes = len(np.unique(y))
        
        SFDA_first = SFDA_model()
        prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)
        # soften the probability using softmax for meaningful confidential mixture
        prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True) 
        means, means_ = SFDA_first._class_means(X, y)  # class means, outer classes means
        # ConfMix
        for y_ in range(num_classes):
            indices = np.where(y == y_)[0]
            y_prob = np.take(prob, indices, axis=0)
            y_prob = y_prob[:, y_]  # probability of correctly classifying x with label y        
            X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                                (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]
        
        SFDA_second = SFDA_model(shrinkage=SFDA_first.shrinkage)
        prob = SFDA_second.fit(X, y).predict_proba(X)   # n * num_cls

        # leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
        sfda_score = np.sum(prob[np.arange(n), y]) / n
        return sfda_score



@njit(cache=True)
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m
# use pseudo data to compile the function
# D = 20, N = 50
# f_tmp = np.random.randn(20, 50).astype(np.float64)
# each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(), 
# np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50, 20)


@njit(cache=True)
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh
# truncated_svd(np.random.randn(20, 10).astype(np.float64))


class LogME(TransferabilityMethod):
    def __init__(self, n_dims=None, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.n_dims = n_dims
        self.reset()
        print('logme initial ')
        f_tmp = np.random.randn(20, 50).astype(np.float64)
        a = each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(), np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50, 20)
        a = truncated_svd(np.random.randn(20, 10).astype(np.float64))
        print(a)


    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        num_classes = constants.num_classes[self.target_dataset]    # [500, 2048]
        labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y # [500, 10]
        return self.fit(self.features, self.y)

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        N, D = f.shape  # k = min(N, D)
        if N > D: # direct SVD may be expensive
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k
        # s.shape = k
        # vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_icml

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        return np.argmax(logits, axis=-1)

