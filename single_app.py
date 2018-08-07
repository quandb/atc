import os
import logging

from atc import flags
from atc.lib import AutomatedTrajectoryClustering

flights_data_flag = flags.create(
    'flights_data',
    flags.FlagType.STRING,
    "Full path to the trajectory file",
    required=True)
logging_file_flag = flags.create(
    'logging_file',
    flags.FlagType.STRING,
    "Full path to the logging file",
    required=False)
lat_column_flag = flags.create(
    'lat_column',
    flags.FlagType.STRING,
    "Latitude column",
    default='Lat')
lon_column_flag = flags.create(
    'lon_column',
    flags.FlagType.STRING,
    "Longitude column",
    default='Lon')
time_column_flag = flags.create(
    'time_column',
    flags.FlagType.STRING,
    "Time column",
    default='TRemains')
source_column_flag = flags.create(
    'source_column',
    flags.FlagType.STRING,
    "Source airport column",
    default='O')
source_airport_flag = flags.create(
    'source_airport',
    flags.FlagType.MULTI_STRING,
    "Source airport column",
    required=True)
des_column_flag = flags.create(
    'des_column',
    flags.FlagType.STRING,
    "Destination airport column",
    default='D')
des_airport_flag = flags.create(
    'des_airport',
    flags.FlagType.MULTI_STRING,
    "Source airport column",
    required=True)
storage_path_flag = flags.create(
    'storage_path',
    flags.FlagType.STRING,
    "Path to folder to store the outcome",
    required=True)
flight_id_column_flag = flags.create(
    'flight_id_column',
    flags.FlagType.STRING,
    "Flight identify column",
    default='ID')
num_points_flag = flags.create(
    'num_points',
    flags.FlagType.INT,
    "Number of points in the curve",
    default=50)
is_plot_flag = flags.create(
    'is_plot',
    flags.FlagType.BOOLEAN,
    "Plotting the routes",
    default=False)
index_flag = flags.create(
    'index',
    flags.FlagType.INT,
    "Validation index for evaluating clustering result",
    default=0)


def run():
    pairs = list(zip(source_airport_flag.value(), des_airport_flag.value()))
    for i, (source_airport, des_airport) in enumerate(pairs):
        params = dict(
            filename=flights_data_flag.value(),
            source_col=source_column_flag.value(),
            des_col=des_column_flag.value(),
            lat_col=lat_column_flag.value(),
            lon_col=lon_column_flag.value(),
            time_col=time_column_flag.value(),
            flight_col=flight_id_column_flag.value(),
            storage_path=storage_path_flag.value(),
            index=index_flag.value(),
        )
        atc_handler = AutomatedTrajectoryClustering(**params)
        # print_progress_bar(i+1, len(source_airport_flag.value()))
        logging.info(
            "\n\nProcess pair (%s - %s)" % (source_airport, des_airport))
        print("\n\nProcess pair (%s - %s)" % (source_airport, des_airport))
        try:
            atc_handler.run(
                source_airport=source_airport,
                des_airport=des_airport,
                num_points=num_points_flag.value(),
                is_plot=is_plot_flag.value(),
            )
        except KeyError as e:
            logging.error(e)
        finally:
            del atc_handler

    logging.info("\n\nDONE")


if __name__ == '__main__':
    flags.parse_flags()
    os.makedirs(storage_path_flag.value(), exist_ok=True)
    log_fname = "%s/monitor.log" % storage_path_flag.value()
    if logging_file_flag.value():
        log_fname = logging_file_flag.value()
    logging.basicConfig(
        filename=log_fname,
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        filemode='w'
    )
    run()
    flags.print_flag_values(storage_path_flag.value())
