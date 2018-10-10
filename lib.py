import logging
import os
from enum import Enum
from itertools import cycle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.distance import directed_hausdorff
from sklearn import preprocessing
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances

from atc import davies_bouldin_index, davies_bouldin_score
from atc.dbcv import DBCV
from atc.frechet import frechetDist
from atc.utils.progress_bar_utils import print_progress_bar


class ValidityIndex(Enum):
  SILHOUETTE = 0
  DAVIES_BOULDIN = 1
  SILHOUETTE_AND_DAVIES_BOULDIN = 2


class AutomatedTrajectoryClustering(object):
  def __init__(self, filename, source_col, des_col, lat_col, lon_col, time_col,
               flight_col, storage_path, index, is_interpolated, is_used_frechet, num_eps_tuning_value):
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
    self.is_interpolated = is_interpolated
    self.is_used_frechet = is_used_frechet
    self.num_eps_tuning_value = num_eps_tuning_value

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
    if self.is_used_frechet:
      return frechetDist(u, v)
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

  def analize_distance_between_terminals(self, flight_data):
    unique_flight_ids = flight_data[self.flight_column].unique()
    distances = []
    for flight_id in unique_flight_ids:
      ter_dis = {}
      ter_dis['flight_id'] = flight_id
      process_flight = flight_data[flight_data[self.flight_column] == flight_id]
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
      is_viz (boolean):

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
    min_sample = int(len(land_flights) / 2)
    # print(terminal_coors)
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
      if not ((mean_lat - 0.5 * std_lat <= lat <= mean_lat + 0.5 * std_lat) and
              (mean_lon - 1 * std_lon <= lon <= mean_lon + 1 * std_lon)):
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
      process_flight = flight_data[flight_data[self.flight_column] == flight_id]
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
    if std_dis < 0.05:
      return []
    abnormal_flights = []
    for item in distances:
      if not mean_dis - 0.5 * std_dis <= item['distance'] <= mean_dis + 1 * std_dis:
        abnormal_flights.append(item['flight_id'])

    return abnormal_flights

  def create_a_home_folder(self, source, des):
    self.storage_path = "%s/%s_%s" % (self.storage_path, source, des)
    os.makedirs(self.storage_path, exist_ok=True)

  def run(self, source_airport, des_airport, num_points, is_plot, k=3, der=0, locker=None):
    start_time = datetime.now()
    self.prefix = "%s-%s: " % (source_airport, des_airport)
    self.__process_data = {}
    self.labels = []
    num_false_interpolate = 0
    if locker:
      with locker:
        flight_df = self.load_data(source_airport, des_airport)
    else:
      flight_df = self.load_data(source_airport, des_airport)
    logging.info(
      self.prefix + self.prefix + "There are %s records for pair (source, des): (%s, %s)" % (
        len(flight_df), source_airport, des_airport))
    flight_ids = flight_df[self.flight_column].unique()

    abnormal_flights = self.detect_abnormal_flight_clustering(flight_df)
    logging.info(self.prefix + "There are %s/%s are outliers removed by DBSCAN." % (
      len(abnormal_flights), len(flight_ids)))
    normal_flights = set(flight_ids) - abnormal_flights
    self.create_a_home_folder(source_airport, des_airport)
    start_interpolate = datetime.now()
    for flight_iden in normal_flights:
      same_flight = flight_df.query("%s == '%s'" % (
        self.flight_column, flight_iden))
      logging.info(self.prefix + "There are %s records for flight: %s" % (
        len(same_flight), flight_iden))
      same_flight = same_flight.drop_duplicates()
      logging.info(self.prefix + "There are %s records after dropping duplicate for flight: %s" % (
        len(same_flight), flight_iden))

      temp_dict = {}
      temp_dict['lat'] = same_flight[self.lat_column]
      temp_dict['lon'] = same_flight[self.lon_column]

      if self.is_interpolated:
        ''' Take interpolate trajectories '''
        adjust_time = same_flight[self.time_column]
        time_sample = self.sampling(adjust_time, num_points)
        interpolate_coor = self.interpolate_data(
          lat=same_flight[self.lat_column],
          lon=same_flight[self.lon_column],
          time=adjust_time,
          time_sample=time_sample,
          der=der,
          k=k)
        temp_dict['inter_lat'] = interpolate_coor['lat']
        temp_dict['inter_lon'] = interpolate_coor['lon']
        if sum(np.isnan(interpolate_coor['lat'])) == 0:
          self.__process_data[flight_iden] = temp_dict
        else:
          num_false_interpolate += 1
          logging.info(self.prefix + "Failed to interpolate %s flight" % flight_iden)
      else:
        ''' Re-sampling '''
        sample_traj = same_flight.sample(num_points - 2).sort_index()
        temp_dict['inter_lat'] = [temp_dict['lat'].iloc[0]] + sample_traj[self.lat_column].tolist() + [temp_dict['lat'].iloc[-1]]
        temp_dict['inter_lon'] = [temp_dict['lon'].iloc[0]] + sample_traj[self.lon_column].tolist() + [temp_dict['lon'].iloc[-1]]
        self.__process_data[flight_iden] = temp_dict

    end_interpolate = datetime.now()
    logging.info(
      self.prefix +
      "There are %s/%s false interpolate flights" % (
        num_false_interpolate, len(flight_ids)))

    ''' Visualize the coordinates '''
    if is_plot:
      self.visualize_original_trajectory(
        source_airport=source_airport, des_airport=des_airport)

    ''' Build the Dissimilarity Matrix '''
    logging.info(self.prefix + "Constructing dissimilarity matrix")
    start_build_distance_mt = datetime.now()
    self.construct_dissimilarity_matrix()
    end_build_distance_mt = datetime.now()

    ''' Perform the clustering with auto tuning parameters '''
    # self.labels = self.route_clustering({})
    logging.info(self.prefix + "Tuning parameter")
    min_samples = self.initialize_min_sample_for_clustering(
      source_airport, des_airport, len(flight_ids))
    start_tuning = datetime.now()
    sil_score, sil_db_score, three_indices_score = self.auto_tuning(
      eps_list=self.sampling(
        self.dissimilarity_matrix[0],
        self.num_eps_tuning_value,
      )[1:],
      min_sample_list=min_samples,
      source=source_airport, des=des_airport)
    end_tuning = datetime.now()
    logging.info(self.prefix + "Start to visualize clusters")
    ''' Clusters viz '''
    self.visualize_detected_clusters(
      source_airport, des_airport,
      sil_score, sil_db_score, three_indices_score)

    ''' Aggregate the clusters '''
    self.visualize_cluster_aggregation(
      source_airport, des_airport,
      sil_score, sil_db_score, three_indices_score)
    end_time = datetime.now()
    end_viz = datetime.now()

    ''' Record time for each step '''
    logging.info("Running time for entire function: %s seconds",
                 (end_time - start_time).seconds)
    logging.info("Interpolate time: %s seconds",
                 (end_interpolate - start_interpolate).seconds)
    logging.info("Build Similarity matrix time: %s seconds",
                 (end_build_distance_mt - start_build_distance_mt).seconds)
    logging.info("Tuning time: %s seconds",
                 (end_tuning - start_tuning).seconds)
    logging.info("Viz time: %s seconds", (end_viz - end_tuning).seconds)

  @classmethod
  def initialize_min_sample_for_clustering(
      cls, source_airport, des_airport, num_of_flights,
      pre_observed=True):
    """
    Initialize min sample value for training clustering model
    Args:
        source_airport (str): Source Airport
        des_airport (str): Destination Airport
        num_of_flights (int): Number of unique flights between OD pair
        pre_observed (boolean): hard values if using flag is true,
            o/w self-adapt to current values

    Returns:
        list[int]: List contains min sample values

    """
    if pre_observed:
      if source_airport == "YBBN" and des_airport == "WSSS":
        return [30]
      if source_airport == "YSSY" and des_airport == "VTBS":
        return [3]
      if source_airport == "NZCH" and des_airport == "WSSS":
        return [2]
    return np.arange(start=5, stop=int(num_of_flights / 2), step=2)

  def visualize_original_trajectory(self, source_airport, des_airport):
    """
    Visualize the original tracks
    Args:
        source_airport (str): Source Airport
        des_airport (str): Destination Airport

    Returns:
        None

    """
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
        self.storage_path, source_airport, des_airport)
    )

  def visualize_detected_clusters(
      self, source_airport, des_airport,
      sil_score, sil_db_score, three_indices_score
  ):
    """
    Visualize clusters that identified by algorithm
    Args:
        source_airport (str): Source Airport
        des_airport (str): Des Airport
        sil_score (float): Score value by Silhouette Index
        sil_db_score (float): Score value by Davies-Boudlin index
        three_indices_score (float): Score value by three indices

    Returns:
        None

    """
    self.cluster_viz(
      labels=self.sil_labels,
      title="Route Clustering for OD pair (%s-%s) based on Silhouette" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s.png" % (
        self.storage_path, source_airport, des_airport,
        'silhouette_%s' % sil_score
      ))
    self.cluster_viz(
      labels=self.sil_db_labels,
      title="Route Clustering for OD pair (%s-%s) based on Silhouette and Davies-Bouldin" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s.png" % (
        self.storage_path, source_airport, des_airport,
        'silhouette_db_%s' % sil_db_score
      ))
    self.cluster_viz(
      labels=self.sil_labels,
      title="Route Clustering for OD pair (%s-%s) based on Three indices" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s.png" % (
        self.storage_path, source_airport, des_airport,
        'three_indices_%s' % three_indices_score
      )
    )

  def visualize_cluster_aggregation(
      self, source_airport, des_airport,
      sil_score, sil_db_score, three_indices_score
  ):
    """
    Visualize aggregated clusters that identified by algorithm
    Args:
        source_airport (str): Source Airport
        des_airport (str): Des Airport
        sil_score (float): Score value by Silhouette Index
        sil_db_score (float): Score value by Davies-Boudlin index
        three_indices_score (float): Score value by three indices

    Returns:
        None

    """
    self.agg_cluster_viz(
      labels=self.sil_labels,
      title="Clusters' Aggregation for OD pair (%s-%s) based on Silhouette" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s_agg.png" % (
        self.storage_path, source_airport, des_airport,
        'silhouette_%s' % sil_score
      )
    )
    self.agg_cluster_viz(
      labels=self.sil_db_labels,
      title="Clusters' Aggregation for OD pair (%s-%s) based on Silhouette and Davies-Bouldin" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s_agg.png" % (
        self.storage_path, source_airport, des_airport,
        'silhouette_db_%s' % sil_db_score
      )
    )
    self.agg_cluster_viz(
      labels=self.three_indices_labels,
      title="Clusters' Aggregation for OD pair (%s-%s) based on Three indices" % (
        source_airport, des_airport),
      pic="%s/%s_%s_%s_agg.png" % (
        self.storage_path, source_airport, des_airport,
        'three_indices_%s' % three_indices_score
      )
    )

  def route_clustering(self, params: dict) -> list:
    clf = DBSCAN(**params, n_jobs=-1)
    return clf.fit_predict(self.dissimilarity_matrix)

  def auto_tuning(self,
                  eps_list, min_sample_list, source, des,
                  tune_plot=True, with_outlier=False):
    tuning_res = []
    tuning_res_append = tuning_res.append
    best_sil_score = -1
    best_sil_db_score = -1
    best_three_indices_score = -1
    for i, eps in enumerate(eps_list):
      print_progress_bar(i + 1, len(eps_list),
                         prefix='Tuning progress', suffix="Processing")
      for min_sample in min_sample_list:
        logging.debug("\t(eps, min_sample): (%s, %s)" % (eps, min_sample))
        params = {'eps': eps,
                  'min_samples': min_sample}
        clustering_params = params.copy()
        clustering_params['metric'] = 'precomputed'
        labels = self.route_clustering(clustering_params)
        unique_clusters = np.unique(labels)

        if len(unique_clusters) is 1:
          continue
        params['#clusters'] = len(unique_clusters)
        logging.info("#clusters detected: %s" % len(unique_clusters))
        logging.debug("\tLabels: %s" % labels)
        filter_dis_matrix = self.dissimilarity_matrix[labels != -1][:, labels != -1]
        filter_labels = [i for i in labels if i != -1]
        params['outlier percentage'] = (len(labels) - len(filter_labels)) * 100. / len(labels)

        ''' Silhouette Scoring '''
        try:
          if with_outlier:
            silhouette_scores = silhouette_samples(
              self.dissimilarity_matrix, labels, metric='precomputed')
            start = 1 if -1 in labels else 0
            ''' Scale to range (0, 1) '''
            params['silhouette_score'] = (np.mean(silhouette_scores[start:]) + 1) / 2.
          else:
            params['silhouette_score'] = (silhouette_score(
              filter_dis_matrix, filter_labels, metric='precomputed') + 1) / 2.
        except ValueError as ve:
          logging.debug(ve)
          continue

        ''' Davies-Bouldin Scoring '''
        if with_outlier:
          db_scores = davies_bouldin_index(
            self.dissimilarity_matrix, labels)
          start = 1 if -1 in labels else 0
          params['db_score'] = min(1, np.mean(db_scores[start:]))
        else:
          params['db_score'] = min(
            1., davies_bouldin_score(filter_dis_matrix, filter_labels))

        ''' DBCV Scoring '''
        params['dbcv_score'] = (DBCV(filter_dis_matrix, filter_labels) + 1) / 2.

        params['silhouette_db_score'] = params['silhouette_score'] + (
            1 - params['db_score'])
        params['three_indices_score'] = params['silhouette_score'] + params['dbcv_score'] + (
            1 - params['db_score'])
        tuning_res_append(params)
        if params['silhouette_score'] > best_sil_score:
          best_sil_score = params['silhouette_score']
          self.sil_labels = labels
        if params['silhouette_db_score'] > best_sil_db_score:
          best_sil_db_score = params['silhouette_db_score']
          self.sil_db_labels = labels
        if params['three_indices_score'] > best_three_indices_score:
          best_three_indices_score = params['three_indices_score']
          self.three_indices_labels = labels
        if tune_plot:
          self.cluster_viz(
            title="Tuning viz for %s clusters with score %s" % (
              len(np.unique(labels)), params['silhouette_score']),
            pic='%s/%s_%s_%s_%sclusters_tuning.png' % (
              self.storage_path, source, des,
              params['silhouette_score'], len(np.unique(labels))),
            labels=labels)

    ''' Store the tuning result '''
    (pd.DataFrame(tuning_res)
      .sort_values(by=['silhouette_score'], ascending=False)
      .to_csv(
      "%s/tuning_result_for_%s_%s_%s.csv" % (
        self.storage_path, source, des, self.index.name),
      index=False))
    return best_sil_score, best_sil_db_score, best_three_indices_score

  def cluster_viz(self, labels, title='', pic='', use_original=True):
    plt.style.use('ggplot')
    colorset = cycle(['purple', 'green', 'red', 'blue', 'orange'])
    for cluster_num in np.unique(labels):
      clr, al = (next(colorset), 1.) if cluster_num != -1 else ('grey', .3)
      for i_flight, run_cluster_number in enumerate(labels):
        if run_cluster_number == cluster_num:
          flight_data = self.__process_data[
            self.le_flight_id.inverse_transform(i_flight)]
          if use_original:
            plt.scatter(
              flight_data['lon'], flight_data['lat'],
              color=clr, marker='o', alpha=al, s=20)
          else:
            plt.scatter(
              flight_data['inter_lon'], flight_data['inter_lat'],
              color=clr, marker='o', alpha=al, s=20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(pic, dpi=500)
    plt.close()

  def agg_cluster_viz(self, labels, title='', pic='', use_original=False):
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
        if use_original:
          lons_append.append(list(flight['lon']))
          lats_append.append(list(flight['lat']))
        else:
          lons_append.append(list(flight['inter_lon']))
          lats_append.append(list(flight['inter_lat']))
      lons_array, lats_array = np.array(lons_append), np.array(lats_append)
      lon = np.mean(lons_array, axis=0)
      lat = np.mean(lats_array, axis=0)
      plt.scatter(
        lon, lat,
        color=clr, marker='o', s=20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(pic, dpi=500)
    plt.close()

  @classmethod
  def coordinate_viz(cls, lats, lons,
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
    # plt.show()
    plt.close()

  @classmethod
  def interpolate_data(cls, lat, lon, time, time_sample, der=0, k=3):
    """
    Transform original data into uniform distribution
    Args:
      lat:
      lon:
      time:
      time_sample:
      der:
      k:

    Returns:

    """
    # print("Time Increase: %s" % time.is_monotonic_increasing)
    # t = time.tolist()
    # for i in range(len(t)-1):
    #   if t[i] > t[i+1]:
    #     print('\t', '%s/%s' % (i, len(t)), t[i], t[i+1])
    # ''' Apply cubic-spline '''
    try:
      ''' lon = f(time) '''
      lon_tck = interpolate.splrep(time.tolist(), lon, s=0, k=k)
      ''' lat = f(time) '''
      lat_tck = interpolate.splrep(time.tolist(), lat, s=0, k=k)
    except ValueError as ve:
      logging.error(ve)
      return {'lat': [np.nan], 'lon': [np.nan]}
    new_lon = interpolate.splev(time_sample, lon_tck, der=der)
    new_lat = interpolate.splev(time_sample, lat_tck, der=der)
    return {'lat': new_lat, 'lon': new_lon}

  @classmethod
  def sampling(cls, values, num_points):
    """
    Scale a vector size into another size
    Args:
      values (list-like): original values will be transforming
      num_points (int): number of points

    Returns:
      list

    """
    # sample_value = signal.resample(values, num_points)
    step_size = (max(values) - min(values)) * 1. / num_points
    sample_value = np.arange(min(values), max(values), step_size)
    return sample_value[:num_points]

  def load_data(self, source_airport, des_airport, deli=','):
    logging.info(
      self.prefix + 'Get start to load data from (%s, %s, %s)' % (
        self.filename, source_airport, des_airport))
    flight_od_df = pd.read_csv(self.filename, delimiter=deli).query(
      "%s == '%s' and %s == '%s'" % (
        self.source_column, source_airport,
        self.des_column, des_airport)
    )
    logging.info(self.prefix + "Load data - #record of (%s, %s): %s" % (
      source_airport, des_airport, len(flight_od_df)))
    flight_od_df = flight_od_df.dropna(axis=0, how='any')
    logging.info(self.prefix + "Filter data - #record of (%s, %s): %s" % (
      source_airport, des_airport, len(flight_od_df)))
    return flight_od_df
