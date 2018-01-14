import pandas as pd
import logging

from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

import flags


flights_data_flag = flags.create(
    'flights_data',
    flags.FlagType.STRING,
    "Full path to the trajectory file",
    required=True)

logging_file_flag = flags.create(
    'logging_file',
    flags.FlagType.STRING,
    "Full path to the logging file",
    required=True)


def load_and_filter_data():
    """
    Load data and perform filtering
    Returns:
      pandas data-frame

    """
    data = pd.read_csv(flights_data_flag.value())
    return data[(data['icao']=='4840D5') & (data['callsign']=='KLM1986_')]
    # return data[(data['icao']=='48520B') & (data['callsign']=='TRA58W__')]

def interpolate_data(data):
    """
    Transform original data into uniform distribution
    Args:
      data:

    Returns:

    """
    ''' Apply cubic-spline '''
    fig, ax = plt.subplots(2)

    sns.regplot(data['lon'], data['lat'], fit_reg=False, ax=ax[0])
    x_new = data.sample(50)['lon']
    tck = interpolate.splrep(
        data['lon'].sort_values(), data['lat'].sort_values(), s=0)
    y_new = interpolate.splev(x_new, tck, der=0)
    print tck
    print y_new
    sns.regplot(x_new, y_new, fit_reg=False, ax=ax[1])
    sns.plt.show()


def perform_dimension_reduction(data):
    """
    Apply PCA to reduce dimension
    Args:
      data:

    Returns:

    """
    pass

def run_clustering(data):
    """
    Run DBSCAN
    Args:
      data: 

    Returns:

    """
    pass


if __name__ == '__main__':
    flags.parse_flags()
    logging.basicConfig(
        filename=logging_file_flag.value(),
        level=logging.DEBUG,
        format='%(asctime)s %(message)s')
    flight_data = load_and_filter_data()
    print "Len: ", len(flight_data)
    print flight_data
    interpolate_data(flight_data)