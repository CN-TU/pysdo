# pysdo
# -----------
# Python implementation of the SDO outlier detection algorithm
#
# Alexander Hartl (alexander.hartl@tuwien.ac.at)
# CN Group, TU Wien
# https://www.cn.tuwien.ac.at/

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.utils import check_array, check_random_state

import math
import numpy as np
import scipy.spatial.distance as distance
import multiprocessing as mp
import ctypes

def _launchParallel(func, n_jobs):
	if n_jobs is None:
		n_jobs = 1
	elif n_jobs < 0:
		n_jobs = mp.cpu_count()
	
	if n_jobs == 1:
		func()
	else:
		processes = []
		for i in range(n_jobs):
			processes.append( mp.Process(target=func))
			processes[-1].start()
		for i in range(n_jobs):
			processes[i].join()

class SDO:
	"""Outlier detection based on Sparse Data Observers"""
	
	def __init__(self, k=None, q=None, qv=0.3, x=6, hbs=False, return_scores = False, contamination=0.1, metric='euclidean', random_state=None, chunksize=1000, n_jobs=None):
		"""
		Parameters
		----------
		k: int, optional
			Number of observers. If None, estimate the number of
			observers using Principal Component Analysis (PCA).
			
		q: int, optional
			Threshold for observed points for model cleaning.
			If None, use qv instead.
			
		qv: float (default=0.3)
			Ratio of removed observers due to model cleaning.
			Only used if q is None.
			
		x: int, optional (default=6)
			Number of nearest observers to consider for outlier scoring
			and model cleaning.
			
		hbs: bool (default=False)
			Whether to use histogram-based sampling.
			
		return_scores: bool (default=False)
			Return outlier scores instead of binary labels.
			
		contamination: float (default=0.1)
			Ratio of outliers in data set. Only used if
			return_scores is False.
			
		metric: string or DistanceMetric (default='euclidean')
			Distance metric to use. Can be a string or distance metric
			as understood by sklearn.neighbors.DistanceMetric.
			
		random_state: RandomState, int or None (default=None)
			If np.RandomState, use random_state directly. If int, use
			random_state as seed value. If None, use default random
			state.
			
		chunksize: int (default=1000)
			Process data in chunks of size chunksize. chunksize has no
			influence on the algorithm's outcome.
			
		n_jobs: int (default=-1)
			Spawn n_jobs threads to process the data. Pass -1 to use as
			many threads as there are CPU cores. n_jobs has no influence
			on the algorithm's outcome.
		"""
		
		if isinstance(k, np.ndarray):
			# ~ self.observers = k.copy()
			self.model = KDTree(k, leaf_size=100, metric=metric)
			self.k = k.shape[0]
		else:
			self.k = k
		self.use_pca = k is None
		self.q = q
		self.qv = qv
		self.x = x
		self.hbs = hbs
		self.return_scores = return_scores
		self.contamination = contamination
		self.metric = metric
		self.random_state = random_state
		self.chunksize = chunksize
		self.n_jobs = n_jobs
		
	def fit(self, X):
		"""
		Train a new model based on X.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
		"""
		
		random = check_random_state(self.random_state)
		X = check_array(X, accept_sparse=['csc'])
		[m, n] = X.shape
		
		if self.use_pca:
			# choose number of observers as described in paper
			pca = PCA()
			pca.fit(X)
			var = max(1,pca.explained_variance_[0])
			sqerr = 0.01 * pca.explained_variance_[0]
			Z = 1.96
			self.k = int((m * Z**2 * var) // ((m-1) * sqerr + Z**2 * var))
			
		if self.hbs:
			Y = X.copy()
			binning_param = 20
			for i in range(n):
				dimMin = min(Y[:,i])
				dimMax = max(Y[:,i])
				if dimMax > dimMin:
					binWidth = (dimMax - dimMin) / round(math.log10(self.k) * binning_param)
					Y[:,i] = (np.floor( (Y[:,i] - dimMin) / binWidth) + .5) * binWidth + dimMin
			Y = np.unique(Y, axis=0)
		else:
			Y = X

		# sample observers randomly from dataset
		observers = Y[random.choice(Y.shape[0], self.k),:]
			
		# copy for efficient cache usage
		#observers = observers.copy()
		model = KDTree(observers, metric=self.metric)
			
		globalI = mp.Value('i', 0)
		
		P = np.frombuffer( mp.Array(ctypes.c_double, self.k).get_obj() )
		P[:] = 0

		def TrainWorker():
			thisP = np.zeros(self.k)
			while True:
				with globalI.get_lock():
					P[:] += thisP[:]
					i = globalI.value
					globalI.value += self.chunksize
				if i >= m: return
				closest = model.query(X[i:(i+self.chunksize)], return_distance=False, k=self.x).flatten()
				#dist = distance.cdist(X[i:(i+self.chunksize)], observers, self.metric)
				#dist_sorted = np.argsort(dist, axis=1)
				#closest = dist_sorted[:,0:self.x].flatten()

				thisP = np.sum (closest[:,None] == np.arange(self.k), axis=0)
				
		_launchParallel(TrainWorker, self.n_jobs)
	
		q = np.quantile(P, self.qv) if self.q is None else self.q
		#self.observers = observers[P>=q].copy()
		self.model = KDTree(observers[P>=q], metric=self.metric)
		
	def predict(self, X):
		"""
		Only perform outlier detection based on a trained model.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
					
		Returns
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for passed input data if return_scores==True,
			otherwise binary outlier labels.
		"""
		
		X = check_array(X, accept_sparse=['csc'])
		[m, n] = X.shape
		
		scores = np.frombuffer( mp.Array(ctypes.c_double, m).get_obj() )
		globalI = mp.Value('i', 0)
		
		def AppWorker():
			while True:
				with globalI.get_lock():
					i = globalI.value
					globalI.value += self.chunksize
				if i >= m: return
				dist_sorted,_ = self.model.query(X[i:(i+self.chunksize)], return_distance=True, k=self.x)
				scores[i:(i+self.chunksize)] = np.median(dist_sorted, axis=1)
				#dist = distance.cdist(X[i:(i+self.chunksize)], self.observers, self.metric)
				#dist_sorted = np.sort(dist, axis=1)
				#scores[i:(i+self.chunksize)] = np.median(dist_sorted[:,0:self.x], axis=1)
				
		_launchParallel(AppWorker, self.n_jobs)

		if self.return_scores:
			return scores
			
		threshold = np.quantile(scores, 1-self.contamination)
		
		return scores > threshold
		
	def fit_predict(self, X):
		"""
		Train a new model based on X and find outliers in X.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
					
		Returns
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for passed input data if return_scores==True,
			otherwise binary outlier labels.
		"""
		self.fit(X)
		return self.predict(X)

	def get_params(self, deep=True):
		"""
		Return the model's parameters

		Parameters
		---------------
		deep : bool, optional (default=True)
			Return sub-models parameters.

		Returns
		---------------
		params: dict, shape (n_parameters,)
			A dictionnary mapping of the model's parameters.
		"""
		return {"k":None if self.use_pca else self.k,
			"q":self.q,
			"qv":self.qv,
			"x":self.x,
			"hbs":self.hbs,
			"return_scores":self.return_scores,
			"contamination":self.contamination,
			"metric":self.metric,
			"random_state":str(self.random_state),
			"chunksize":self.chunksize,
			"n_jobs":self.n_jobs}
