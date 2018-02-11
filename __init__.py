# See: http://docs.python.org/library/pkgutil.html#pkgutil.extend_path
from pkgutil import extend_path

from sklearn.metrics.cluster.unsupervised import check_number_of_labels
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing.label import LabelEncoder
from sklearn.utils.validation import check_X_y

__path__ = extend_path(__path__, __name__)
import numpy as np


def davies_bouldin_index(X, labels, metric='euclidean'):
  """Compute the Davies Bouldin index.
  The index is defined as the ratio of within-cluster
  and between-cluster distances.
  Parameters
  ----------
  X : array-like, shape (``n_samples``, ``n_features``)
      List of ``n_features``-dimensional data points. Each row corresponds
      to a single data point.
  labels : array-like, shape (``n_samples``,)
      Predicted labels for each sample.
  Returns
  -------
  score : float
      The resulting Davies-Bouldin index.
  References
  ----------
  .. [1] `Davies, David L.; Bouldin, Donald W. (1979).
     "A Cluster Separation Measure". IEEE Transactions on
     Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227`_
  """
  X, labels = check_X_y(X, labels)
  le = LabelEncoder()
  labels = le.fit_transform(labels)
  n_samples, _ = X.shape
  n_labels = len(le.classes_)

  check_number_of_labels(n_labels, n_samples)
  intra_dists = np.zeros(n_labels)
  centroids = np.zeros((n_labels, len(X[0])), np.float32)
  # print("Start")
  # print(labels)
  # print(X)
  for k in range(n_labels):
    cluster_k = X[labels == k]
    mean_k = np.mean(cluster_k, axis=0)
    centroids[k] = mean_k
    # print("Process")
    # print(mean_k)
    # print(cluster_k)
    intra_dists[k] = np.average(
      pairwise_distances(
        cluster_k, [mean_k], metric=metric))
  centroid_distances = pairwise_distances(centroids, metric=metric
                                          )
  with np.errstate(divide='ignore', invalid='ignore'):
    if np.all((intra_dists[:, None] + intra_dists) == 0.0) or \
      np.all(centroid_distances == 0.0):
        return 0.0
    scores = (intra_dists[:, None] + intra_dists)/centroid_distances
    # remove inf values
    scores[scores == np.inf] = np.nan
    return np.nanmax(scores, axis=1)


def davies_bouldin_score(X, labels, metric='euclidean'):
  """Compute the Davies Bouldin index.
  The index is defined as the ratio of within-cluster
  and between-cluster distances.
  Parameters
  ----------
  X : array-like, shape (``n_samples``, ``n_features``)
      List of ``n_features``-dimensional data points. Each row corresponds
      to a single data point.
  labels : array-like, shape (``n_samples``,)
      Predicted labels for each sample.
  Returns
  -------
  score : float
      The resulting Davies-Bouldin index.
  References
  ----------
  .. [1] `Davies, David L.; Bouldin, Donald W. (1979).
     "A Cluster Separation Measure". IEEE Transactions on
     Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227`_
  """
  return np.mean(davies_bouldin_index(X, labels, metric))
