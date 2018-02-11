import logging
from enum import Enum
from itertools import cycle

import pandas as pd
from pandas.stats.moments import rolling_median
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from sklearn import preprocessing
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.externals.joblib import Memory
from sklearn.metrics import silhouette_score, silhouette_samples
# import hdbscan

from atc.frechet import frechetDist
from atc import davies_bouldin_index, davies_bouldin_score
from sklearn.metrics.cluster.unsupervised import calinski_harabaz_score
from sklearn.metrics.pairwise import euclidean_distances


class ValidityIndex(Enum):
  SILHOUETTE = 0
  DAVIES_BOULDIN = 1
  SILHOUETTE_AND_DAVIES_BOULDIN = 2


class AutomatedTrajectoryClustering(object):
  def __init__(self, filename, source_col, des_col, lat_col, lon_col, time_col,
               flight_col, storage_path, index):
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
    self.index = ValidityIndex(index)

  def construct_dissimilarity_matrix(self):
    self.le_flight_id = preprocessing.LabelEncoder()
    self.le_flight_id.fit(list(self.__process_data.keys()))
    num_flights = len(self.__process_data)
    self.dissimilarity_matrix = np.ndarray(
      shape=(num_flights, num_flights),
      dtype=float)
    for i in range(num_flights):
      for j in range(i, num_flights):
        if i != j:
          from_ = self.__process_data[self.le_flight_id.inverse_transform(i)]
          to_ = self.__process_data[self.le_flight_id.inverse_transform(j)]
          distance = self.compute_the_distance(
            u=list(zip(from_['inter_lon'], from_['inter_lat'])),
            v=list(zip(to_['inter_lon'], to_['inter_lat'])))
        else:
          distance = 0
        self.dissimilarity_matrix[i, j] = distance
        self.dissimilarity_matrix[j, i] = distance

  def compute_the_distance(self, u, v):
    return frechetDist(u, v)
    # return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

  def analize_distance_between_terminals(self, flight_data):
    unique_flight_ids = flight_data[self.flight_column].unique()
    distances = []
    for flight_id in unique_flight_ids:
      ter_dis = {}
      ter_dis['flight_id'] = flight_id
      process_flight = flight_data[flight_data[self.flight_column]==flight_id]
      lats = list(process_flight[self.lat_column])
      lons = list(process_flight[self.lon_column])
      ter_dis['depart'] = (lons[0], lats[0])
      ter_dis['land'] = (lons[-1], lats[-1])
      ter_dis['distance'] = euclidean_distances(
        [ter_dis['depart']], [ter_dis['land']])[0][0]
      distances.append(ter_dis)
    just_dis = [i['distance'] for i in distances]
    print("Describe", pd.Series(just_dis).describe())
    print("Values: ", just_dis)

  def sum_the_ternimal_trajectories(self, flight_data):
    depart_flight = flight_data.drop_duplicates(
      subset=self.flight_column, keep='first')
    plt.scatter(depart_flight[self.lon_column], depart_flight[self.lat_column])
    plt.show()
    land_flight = flight_data.drop_duplicates(
      subset=self.flight_column, keep='last')
    plt.scatter(land_flight[self.lon_column], land_flight[self.lat_column])
    plt.show()
    print("Depart Lat Sum:", depart_flight[self.lat_column].describe())
    print("Depart Lon Sum:", depart_flight[self.lon_column].describe())
    print("Landing Lat Sum:", land_flight[self.lat_column].describe())
    print("Landing Lon Sum:", land_flight[self.lon_column].describe())

  def detect_abnormal_flight_clustering(self, flight_data, is_viz=False):
    """
    Detect flights that depart and land out of terminal
    Args:
      flight_data (pd DataFrame): flight data from source to des
      keep (str): keep must be either 'first' or 'last'

    Returns:
      pd DataFrame: filtered flight data

    """
    land_flights = flight_data.drop_duplicates(
      subset=self.flight_column, keep='last')
    depart_flights = flight_data.drop_duplicates(
      subset=self.flight_column, keep='first')
    terminal_flights = land_flights.append(depart_flights)
    terminal_coors = terminal_flights[
      [self.lon_column, self.lat_column]].as_matrix()
    min_sample = int(len(land_flights)/2)
    labels = DBSCAN(
      min_samples=min_sample,
      n_jobs=-1).fit_predict(terminal_coors)
    outlier_flights = set(terminal_flights[labels == -1][self.flight_column])

    ''' Viz '''
    if is_viz:
      plt.style.use('ggplot')
      plt.scatter(
        x=terminal_coors[labels != -1][:, 0],
        y=terminal_coors[labels != -1][:, 1],
        marker='o', s=10, c='blue')
      plt.scatter(
        x=terminal_coors[labels == -1][:, 0],
        y=terminal_coors[labels == -1][:, 1],
        marker='o', s=10, c='red')
      plt.xlabel("Longitude")
      plt.ylabel("Latitude")
      plt.title("Outlier clustering detect %s/%s outliers" % (
        len(outlier_flights),
        len(land_flights)))
      fig = plt.gcf()
      fig.set_size_inches((11, 8.5), forward=False)
      # fig.savefig(pic_name, dpi=500)
      # plt.close()
      plt.show()

    return outlier_flights

  def detect_abnormal_flight_by_coor(self, flight_data, keep):
    """
    Detect flights that depart and land out of terminal
    Args:
      flight_data (pd DataFrame): flight data from source to des
      keep (str): keep must be either 'first' or 'last'

    Returns:
      pd DataFrame: filtered flight data

    """
    assert keep in ['first', 'last']
    unique_flights = flight_data.drop_duplicates(
      subset=self.flight_column, keep=keep)
    mean_lat = unique_flights[self.lat_column].mean()
    std_lat = unique_flights[self.lat_column].std()
    mean_lon = unique_flights[self.lon_column].mean()
    std_lon = unique_flights[self.lon_column].std()
    ''' No abnormal with small std '''
    if std_lat < 0.05 and std_lon < 0.05:
      return []
    abnormal_flights = []
    for lat, lon, flight_id in zip(unique_flights[self.lat_column],
                                   unique_flights[self.lon_column],
                                   unique_flights[self.flight_column]):
      if not ((mean_lat - 0.5*std_lat <= lat <= mean_lat + 0.5*std_lat) and
         (mean_lon - 1*std_lon <= lon <= mean_lon + 1*std_lon)):
        abnormal_flights.append(flight_id)
    return abnormal_flights

  def detect_abnormal_flight_by_distances(self, flight_data):
    """
    Detect flights that depart and land out of terminal
    Args:
      flight_data (pd DataFrame): flight data from source to des

    Returns:
      pd DataFrame: filtered flight data

    """
    unique_flight_ids = flight_data[self.flight_column].unique()
    distances = []
    for flight_id in unique_flight_ids:
      ter_dis = {}
      ter_dis['flight_id'] = flight_id
      process_flight = flight_data[flight_data[self.flight_column]==flight_id]
      lats = list(process_flight[self.lat_column])
      lons = list(process_flight[self.lon_column])
      ter_dis['depart'] = (lons[0], lats[0])
      ter_dis['land'] = (lons[-1], lats[-1])
      ter_dis['distance'] = euclidean_distances(
        [ter_dis['depart']], [ter_dis['land']])[0][0]
      distances.append(ter_dis)
    just_dis = pd.Series([i['distance'] for i in distances])
    mean_dis = just_dis.mean()
    std_dis = just_dis.std()
    # plt.boxplot(just_dis)
    # plt.show()
    threshold = 5
    print("Describe", pd.Series(just_dis).describe())
    print("Accepted range: (%s, %s)" % (
      mean_dis - 0.5*std_dis, mean_dis + 1*std_dis))
    # print("Quantile", just_dis.quantile(0.25))
    # t = rolling_median(just_dis, window=1, center=True).fillna(method='bfill').fillna(method='ffill')
    # difference = np.abs(just_dis - t)
    # outlier_idx = difference > threshold
    # figsize = (7, 2.75)
    # kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)
    # fig, ax = plt.subplots(figsize=figsize)
    # just_dis.plot()
    # just_dis[outlier_idx].plot(**kw)
    # # pd.Series(t).plot(color='r')
    # _ = ax.set_ylim(-50, 50)
    # plt.show()
    # print("Values: ", t)
    if std_dis < 0.05:
      return []
    abnormal_flights = []
    for item in distances:
      # if item['flight_id'] in ['UAE334-1486966800-schedule-0000']:
      #   print("\tHere: ", item['distance'])
      if not mean_dis - 0.5*std_dis <= item['distance'] <= mean_dis + 1*std_dis:
      # if not item['distance'] <= mean_dis + 1*std_dis:
        abnormal_flights.append(item['flight_id'])
        # print(item['distance'])

    return abnormal_flights

  def run(self, source_airport, des_airport, num_points, is_plot, k=3, der=0):
    self.__process_data = {}
    self.labels = []
    num_false_interpolate = 0
    flight_df = self.load_data(source_airport, des_airport)
    # self.analize_distance_between_terminals(flight_df)
    # return None
    logging.info("There are %s records for pair (source, des): (%s, %s)" % (
      len(flight_df), source_airport, des_airport))
    flight_ids = flight_df[self.flight_column].unique()
    # coor_abnormal_flights = set(
    #   self.detect_abnormal_flight_by_coor(flight_df, keep='first')).union(
    #     self.detect_abnormal_flight_by_coor(flight_df, keep='last'))
    # logging.info("There are %s/%s are outliers by coordinate removal." % (
    #   len(coor_abnormal_flights), len(flight_ids)))
    #
    # abnormal_flights = set(self.detect_abnormal_flight_by_distances(flight_df))
    # logging.info("There are %s/%s are outliers by distance removal." % (
    #   len(abnormal_flights), len(flight_ids)))
    # logging.info(
    #   "coor - dis removal: %s", coor_abnormal_flights - abnormal_flights)
    # logging.info(
    #   "dis - coor removal: %s", abnormal_flights - coor_abnormal_flights)

    abnormal_flights = self.detect_abnormal_flight_clustering(flight_df)
    logging.info("There are %s/%s are outliers removed by DBSCAN." % (
      len(abnormal_flights), len(flight_ids)))
    normal_flights = set(flight_ids) - abnormal_flights
    for flight_iden in normal_flights:
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
        time_sample=time_sample,
        der=der,
        k=k)
      temp_dict = {}
      temp_dict['lat'] = same_flight[self.lat_column]
      temp_dict['lon'] = same_flight[self.lon_column]
      temp_dict['inter_lat'] = interpolate_coor['lat']
      temp_dict['inter_lon'] = interpolate_coor['lon']
      # temp_dict['inter_lat'] = temp_dict['lat']
      # temp_dict['inter_lon'] = temp_dict['lon']
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
        title="Original Coordinate for %s - %s" % (source_airport, des_airport),
        pic_name="%s/%s_%s_original_coordinates.png" % (
          self.storage_path, source_airport, des_airport))
      self.coordinate_viz(
        lats=(v['inter_lat'] for v in self.__process_data.values()),
        lons=(v['inter_lon'] for v in self.__process_data.values()),
        title="Interpolated Coordinate %s - %s" % (source_airport, des_airport),
        pic_name="%s/%s_%s_interpolated_coordinates.png" % (
          self.storage_path, source_airport, des_airport),
        # is_colornized=True
        )

    ''' Build the Dissimilarity Matrix '''
    logging.info("Constructing dissimilarity matrix")
    self.construct_dissimilarity_matrix()

    ''' Perform the clustering with auto tuning parameters '''
    # self.labels = self.route_clustering({})
    logging.info("Tuning parameter")
    # logging.info("Distance: %s" % self.dissimilarity_matrix[0])
    best_score = self.auto_tuning(
      eps_list=self.sampling(self.dissimilarity_matrix[0], 100)[1:],
      min_sample_list=np.arange(start=2, stop=int(len(flight_ids)/2), step=2),
      # eps_list=[1, 3],
      # min_sample_list=[3, 5],
      soure=source_airport, des=des_airport)
    logging.info("Best score is %s" % best_score)
    ''' Clusters viz '''
    self.cluster_viz(
      labels=self.labels,
      title="Route Clustering for OD pair (%s-%s)" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s.png" % (
        self.storage_path,source_airport, des_airport, best_score
      ))
    self.agg_cluster_viz(
      labels=self.labels,
      title="Clusters' Aggregation for OD pair (%s-%s)" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s_agg.png" % (
        self.storage_path,source_airport, des_airport, best_score
      ))

  def route_clustering(self, params: dict) -> list:
    clf = DBSCAN(**params, n_jobs=-1)
    # clf = hdbscan.HDBSCAN(
    #   algorithm='best', alpha=1.0, approx_min_span_tree=True,
    #   gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
    #   metric=params['metric'], min_cluster_size=params['eps'],
    #   min_samples=params['min_samples'],
    #   p=None)
    return clf.fit_predict(self.dissimilarity_matrix)

  def auto_tuning(self,
                  eps_list, min_sample_list, soure, des,
                  tune_plot=True, with_outlier=False):
    tuning_res = []
    tuning_res_append = tuning_res.append
    best_score = -1
    for eps in eps_list:
      for min_sample in min_sample_list:
        logging.debug("(eps, min_sample): (%s, %s)" % (eps, min_sample))
        params = {'eps': eps,
                  'min_samples': min_sample}
        clustering_params = params.copy()
        clustering_params['metric'] = 'precomputed'
        labels = self.route_clustering(clustering_params)
        unique_clusters = np.unique(labels)
        if len(unique_clusters) is 1:
          continue
        params['#clusters'] = len(unique_clusters)
        logging.debug("\tLabels: %s" % labels)
        filter_dis_matrix = self.dissimilarity_matrix[labels != -1][:, labels != -1]
        filter_labels = [i for i in labels if i != -1]
        params['outlier percentage'] = (
            len(labels) - len(filter_labels))*100./len(labels)
        try:
          if with_outlier:
            silhouette_scores = silhouette_samples(
              self.dissimilarity_matrix, labels, metric='precomputed')
            start = 1 if -1 in labels else 0
            ''' Scale to range (0, 1) '''
            params['silhouette_score'] = (
              np.mean(silhouette_scores[start:]) + 1)/2.
          else:
            params['silhouette_score'] = (silhouette_score(
              filter_dis_matrix, filter_labels, metric='precomputed') + 1)/2.
        except ValueError as ve:
          logging.error(ve)
          continue

        if with_outlier:
          db_scores = davies_bouldin_index(
            self.dissimilarity_matrix, labels)
          start = 1 if -1 in labels else 0
          params['db_score'] = min(1, np.mean(db_scores[start:]))
        else:
          params['db_score'] = min(1, davies_bouldin_score(filter_dis_matrix, filter_labels,))
        params['silhouette_db_score'] = params['silhouette_score'] + (
          1 - params['db_score'])
        tuning_res_append(params)
        use_score = params['silhouette_score']
        if self.index is ValidityIndex.DAVIES_BOULDIN:
          use_score = params['db_score']
        elif self.index is ValidityIndex.SILHOUETTE_AND_DAVIES_BOULDIN:
          use_score = params['silhouette_db_score']
        if use_score > best_score:
          best_score = use_score
          self.labels = labels
        if tune_plot:
          self.cluster_viz(
            title="Tuning viz for %s clusters with score %s" % (
              len(np.unique(labels)), params['silhouette_score']),
            pic='%s/%s_%s_%sclusters_%s_tuning.png' % (
              self.storage_path, soure, des,
              len(np.unique(labels)), params['silhouette_score']),
            labels=labels)

    ''' Store the tuning result '''
    (pd.DataFrame(tuning_res)
      .sort_values(by=['silhouette_score'], ascending=False)
      .to_csv(
        "%s/tuning_result_for_%s_%s_%s.csv" % (
          self.storage_path, soure, des, self.index.name),
        index=False))
    return best_score

  def cluster_viz(self, labels, title='', pic=''):
    plt.style.use('ggplot')
    colorset = cycle(['purple', 'green', 'red', 'blue', 'orange'])
    for cluster_num in np.unique(labels):
      clr, al = (next(colorset), 1.) if cluster_num != -1 else ('grey', .3)
      for i_flight, run_cluster_number in enumerate(labels):
        if run_cluster_number == cluster_num:
          flight_data = self.__process_data[
            self.le_flight_id.inverse_transform(i_flight)]
          plt.scatter(
            flight_data['inter_lon'], flight_data['inter_lat'],
            color=clr, marker='o', alpha=al, s=20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    # plt.savefig(pic, dpi=100)
    fig = plt.gcf()
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(pic, dpi=500)
    plt.close()

  def agg_cluster_viz(self, labels, title='', pic=''):
    plt.style.use('ggplot')
    colorset = cycle(['purple', 'green', 'red', 'blue', 'orange'])
    for cluster_num in set(labels) - set([-1]):
      clr = next(colorset)
      i_flights = pd.DataFrame(
        self.dissimilarity_matrix)[labels == cluster_num].index.values
      lons_append = []
      lats_append = []
      for i in i_flights:
        flight = self.__process_data[self.le_flight_id.inverse_transform(i)]
        lons_append.append(flight['inter_lon'])
        lats_append.append(flight['inter_lat'])
        # print("Inter len:", len(flight['inter_lon']), len(flight['inter_lat']))
      lons_array, lats_array = np.array(lons_append), np.array(lats_append)
      # lon_mean = [np.mean(lons_array[:, i_col])
      #             for i_col in range(len(lons_append[0]))]
      # lat_mean = [np.mean(lats_array[:, i_col])
      #             for i_col in range(len(lats_append[0]))]
      # print("Cluster", cluster_num)
      # print(len(lons_append[0]), len(lats_append[0]))
      # print(lons_append[0], '\n', lats_append[0])
      # print(len(lons_array[0]), len(lats_array[0]))
      lon = np.mean(lons_array, axis=0)
      lat = np.mean(lats_array, axis=0)
      # print(len(lon), lon)
      # print(len(lat), lat)
      plt.scatter(
        lon, lat,
        color=clr, marker='o', s=20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    # plt.savefig(pic, dpi=100)
    fig = plt.gcf()
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(pic, dpi=500)
    plt.close()

  def coordinate_viz(self, lats, lons,
                     title='', pic_name='', is_colornized=False):
    plt.style.use('ggplot')
    if is_colornized:
      colorset = cycle(['purple', 'green', 'red', 'blue', 'orange'])
    else:
      colorset = cycle(['blue'])
    for color, lat, lon in zip(colorset, lats, lons):
      plt.scatter(x=lon, y=lat, marker='o', s=10, c=color)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(pic_name, dpi=500)
    plt.close()

  def interpolate_data(self, lat, lon, time, time_sample, der=0, k=3):
    """
    Transform original data into uniform distribution
    Args:
      data:

    Returns:

    """
    # print("Time Decrease: %s" % time.is_monotonic_decreasing)
    ''' Apply cubic-spline '''
    ''' lon = f(time) '''
    lon_tck = interpolate.splrep(time, lon, s=0, k=k)
    ''' lat = f(time) '''
    lat_tck = interpolate.splrep(time, lat, s=0, k=k)
    new_lon = interpolate.splev(time_sample, lon_tck, der=der)
    new_lat = interpolate.splev(time_sample, lat_tck, der=der)
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
    return sample_value[:num_points]

  def load_data(self, source_airport, des_airport):
    logging.info('Get start to load data from (%s, %s, %s)' % (
      self.filename, source_airport, des_airport))
    # print("CROSS")
    # t_df = pd.read_csv(self.filename, delimiter='\t')
    # print(pd.crosstab(t_df[self.source_column], t_df[self.des_column]))
    return pd.read_csv(self.filename, delimiter='\t').query(
      "%s == '%s' and %s == '%s'" % (
        self.source_column, source_airport,
        self.des_column, des_airport))

