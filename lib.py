import logging
from itertools import cycle

import pandas as pd
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from sklearn import preprocessing
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

from atc.frechet import frechetDist


class AutomatedTrajectoryClustering(object):
  def __init__(self, filename, source_col, des_col, lat_col, lon_col, time_col,
               flight_col, storage_path):
    self.filename = filename
    self.source_column = source_col
    self.des_column = des_col
    self.lat_column = lat_col
    self.lon_column = lon_col
    self.time_column = time_col
    self.flight_column = flight_col
    self.storage_path = storage_path
    self.__process_data = {}
    self.labels = []

  def construct_dissimilarity_matrix(self):
    self.le_flight_id = preprocessing.LabelEncoder()
    self.le_flight_id.fit(list(self.__process_data.keys()))
    num_flights = len(self.__process_data)
    self.dissimilarity_matrix = np.ndarray(
      shape=(num_flights, num_flights),
      dtype=float)
    for i in range(num_flights):
      for j in range(num_flights):
        from_ = self.__process_data[self.le_flight_id.inverse_transform(i)]
        to_ = self.__process_data[self.le_flight_id.inverse_transform(j)]
        # print "Flight ID: ", self.le_flight_id.inverse_transform(i)
        # print from_['inter_lat']
        self.dissimilarity_matrix[i, j] = self.compute_the_distance(
          u=list(zip(from_['inter_lon'], from_['inter_lat'])),
          v=list(zip(to_['inter_lon'], to_['inter_lat'])))

  def compute_the_distance(self, u, v):
    return frechetDist(u, v)
    # return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

  def run(self, source_airport, des_airport, num_points, is_plot):
    num_false_interpolate = 0
    flight_df = self.load_data(source_airport, des_airport)
    logging.info("There are %s records for pair (source, des): (%s, %s)" % (
      len(flight_df), source_airport, des_airport))

    flight_ids = flight_df[self.flight_column].unique()
    for flight_iden in flight_ids:

      same_flight = flight_df.query("%s == '%s'" % (
        self.flight_column, flight_iden))[::-1]
      logging.info("There are %s records for flight: %s" % (
        len(same_flight), flight_iden))
      adjust_time = same_flight[self.time_column]
      time_sample = self.sampling(adjust_time, num_points)
      interpolate_coor = self.interpolate_data(
        lat=same_flight[self.lat_column],
        lon=same_flight[self.lon_column],
        time=adjust_time,
        time_sample=time_sample
      )
      temp_dict = {}
      temp_dict['lat'] = same_flight[self.lat_column]
      temp_dict['lon'] = same_flight[self.lon_column]
      temp_dict['inter_lat'] = interpolate_coor['lat']
      temp_dict['inter_lon'] = interpolate_coor['lon']
      if sum(np.isnan(interpolate_coor['lat'])) == 0:
        self.__process_data[flight_iden] = temp_dict
      else:
        num_false_interpolate += 1
        logging.info("Failed to interpolate %s flight" % flight_iden)

    logging.info(
      "There are %s/%s false interpolate flights" % (
        num_false_interpolate, len(flight_ids)))
    ''' Visualize the coordinates '''
    if is_plot:
      self.coordinate_viz(
        lats=(v['lat'] for v in self.__process_data.values()),
        lons=(v['lon'] for v in self.__process_data.values()),
        title="Original Coordinate")
      self.coordinate_viz(
        lats=(v['inter_lat'] for v in self.__process_data.values()),
        lons=(v['inter_lon'] for v in self.__process_data.values()),
        title="Interpolated Coordinate")

    ''' Build the Dissimilarity Matrix '''
    logging.info("Constructing dissimilarity matrix")
    self.construct_dissimilarity_matrix()

    ''' Perform the clustering with auto tuning parameters '''
    self.auto_tuning(
      eps_list=np.append(
        self.sampling(self.dissimilarity_matrix[0], 50)[1:10], [3]),
      min_sample_list=[3, 4, 5, 9, 16, 25, 36],
      soure=source_airport, des=des_airport)

    ''' Clusters viz '''
    self.cluster_viz(title="Route Clustering for OD pair (%s-%s)" % (
      source_airport, des_airport))

  def route_clustering(self, params: dict) -> list:
    clf = DBSCAN(**params)
    return clf.fit_predict(self.dissimilarity_matrix)

  def auto_tuning(self, eps_list, min_sample_list, soure, des):
    tuning_res = []
    tuning_res_append = tuning_res.append
    best_silhouette = -1
    for eps in eps_list:
      for min_sample in min_sample_list:
        logging.debug("(eps, min_sample): (%s, %s)" % (eps, min_sample))
        params = {'eps': eps,
                  'min_samples': min_sample,
                  'metric': 'precomputed'}
        labels = self.route_clustering(params)
        params['#clusters'] = np.unique(labels)
        logging.debug("\tLabels: %s" % labels)
        try:
          params['silhouette_score'] = silhouette_score(
            self.dissimilarity_matrix, labels, metric='precomputed')
        except ValueError as ve:
          logging.error(ve)
          params['silhouette_score'] = -2
        tuning_res_append(params)
        if params['silhouette_score'] > best_silhouette:
          best_silhouette = params['silhouette_score']
          self.labels = labels

    ''' Store the tuning result '''
    (pd.DataFrame(tuning_res)
      .sort_values(by=['silhouette_score'], ascending=False)
      .to_csv(
        "%s/tuning_result_for_%s_%s.csv" % (self.storage_path, soure, des),
        index=False))

  def cluster_viz(self, title='', pic=''):
    plt.style.use('ggplot')
    colorset = cycle(['purple', 'green', 'red', 'blue', 'orange'])
    for cluster_num in np.unique(self.labels):
      clr, al = (next(colorset), 1.) if cluster_num != -1 else ('grey', .3)
      for i_flight, run_cluster_number in enumerate(self.labels):
        if run_cluster_number == cluster_num:
          flight_data = self.__process_data[
            self.le_flight_id.inverse_transform(i_flight)]
          plt.scatter(
            flight_data['inter_lon'], flight_data['inter_lat'],
            color=clr, marker='o', alpha=al, s=20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()

  def coordinate_viz(self, lats, lons, title='', pic_name=''):
    plt.style.use('ggplot')
    for lat, lon in zip(lats, lons):
      plt.scatter(x=lon, y=lat, marker='o', s=10, c='b')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()

  def interpolate_data(self, lat, lon, time, time_sample):
    """
    Transform original data into uniform distribution
    Args:
      data:

    Returns:

    """
    # print("Time Decrease: %s" % time.is_monotonic_decreasing)
    ''' Apply cubic-spline '''
    ''' lon = f(time) '''
    lon_tck = interpolate.splrep(time, lon, s=0)
    ''' lat = f(time) '''
    lat_tck = interpolate.splrep(time, lat, s=0)
    new_lon = interpolate.splev(time_sample, lon_tck, der=0)
    new_lat = interpolate.splev(time_sample, lat_tck, der=0)
    return {'lat': new_lat, 'lon': new_lon}

  def sampling(self, values, num_points):
    """
    Scale a vector size into another size
    Args:
      values (list-like): original values will be transforming
      num_points (int): number of points

    Returns:
      list

    """
    # sample_value = signal.resample(values, num_points)
    step_size = (max(values) - min(values))*1./num_points
    sample_value = np.arange(min(values), max(values), step_size)
    return sample_value

  def load_data(self, source_airport, des_airport):
    logging.info('Get start to load data from (%s, %s, %s)' % (
      self.filename, source_airport, des_airport))
    return pd.read_csv(self.filename, delimiter='\t').query(
      "%s == '%s' and %s == '%s'" % (
        self.source_column, source_airport,
        self.des_column, des_airport))

