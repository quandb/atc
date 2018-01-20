import logging
from itertools import cycle

import pandas as pd
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from sklearn import preprocessing
from sklearn.cluster.dbscan_ import DBSCAN


class AutomatedTrajectoryClustering(object):
  def __init__(self, filename, source_col, des_col, lat_col, lon_col, time_col,
               flight_col):
    self.filename = filename
    self.source_column = source_col
    self.des_column = des_col
    self.lat_column = lat_col
    self.lon_column = lon_col
    self.time_column = time_col
    self.flight_column = flight_col
    self.__process_data = {}

  def construct_dissimilarity_matrix(self):
    self.le_flight_id = preprocessing.LabelEncoder()
    self.le_flight_id.fit(self.__process_data.keys())
    num_flights = len(self.__process_data)
    self.dissimilarity_matrix = np.ndarray(
      shape=(num_flights,num_flights),
      dtype=float)
    for i in xrange(num_flights):
      for j in xrange(num_flights):
        from_ = self.__process_data[self.le_flight_id.inverse_transform(i)]
        to_ = self.__process_data[self.le_flight_id.inverse_transform(j)]
        # print "Flight ID: ", self.le_flight_id.inverse_transform(i)
        # print from_['inter_lat']
        self.dissimilarity_matrix[i, j] = self.compute_the_distance(
          u=zip(from_['inter_lon'], from_['inter_lat']),
          v=zip(to_['inter_lon'], to_['inter_lat']))

  def compute_the_distance(self, u, v):
    return directed_hausdorff(u, v)[0]

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

    logging.info(
      "There are %s false interpolate flights" % num_false_interpolate)
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
    self.construct_dissimilarity_matrix()

    ''' Perform the clustering '''
    self.route_clustering()

    ''' Clusters viz '''
    self.cluster_viz(title="Route Clustering")

  def route_clustering(self):
    clf = DBSCAN(eps=5, min_samples=3)
    self.labels = clf.fit_predict(self.dissimilarity_matrix)
    print("LABELS for %s routes" % len(self.dissimilarity_matrix))

  def cluster_viz(self, title='', pic=''):
    plt.style.use('ggplot')
    colorset = cycle(['purple', 'green', 'red', 'blue', 'orange'])
    for cluster_num in np.unique(self.labels):
      clr, mar = (colorset.next(), 'o') if cluster_num != -1 else ('black', '.')
      for i_flight, run_cluster_number in enumerate(self.labels):
        if run_cluster_number == cluster_num:
          flight_data = self.__process_data[
            self.le_flight_id.inverse_transform(i_flight)]
          plt.scatter(
            flight_data['inter_lon'], flight_data['inter_lat'],
            color=clr, marker=mar, alpha=1.0)
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
    step_size = (values.max() - values.min())*1./num_points
    sample_value = np.arange(values.min(), values.max(), step_size)
    return sample_value

  def load_data(self, source_airport, des_airport):
    logging.info('Get start to load data from (%s, %s, %s)' % (
      self.filename, source_airport, des_airport))
    return pd.read_csv(self.filename, delimiter='\t').query(
      "%s == '%s' and %s == '%s'" % (
        self.source_column, source_airport,
        self.des_column, des_airport))

