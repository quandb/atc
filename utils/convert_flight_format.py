from copy import deepcopy

import pandas as pd
import json

from atc import flags
from atc.utils.progress_bar_utils import print_progress_bar

flights_data_flag = flags.create(
    'flights_data',
    flags.FlagType.STRING,
    "Full path to the trajectory file in json format",
    required=True)


if __name__ == '__main__':
    flags.parse_flags()
    print("Filepath :", flights_data_flag.value())
    flights = []
    flight_append = flights.append
    with open(flights_data_flag.value()) as fin:
        for line in fin:
            flight_keys = [
                "Flight ID", "Ident", "Origin", "Destination",
                "Actual Arrival Time (UTC)"]
            tract_keys = [
                "Time (UTC)",
                "Latitude",
                "Longitude",
                # "Altitude (ft)",
                # "Rate",
                # "Course",
                # "Direction",
                # "Facility Name",
                # "Facility Description",
                # "Estimated Pos.",
                "TTravelled",
                "TRemains",
                "DRemains"
            ]
            one_flight = (json.loads(line))
            flight_header = {}
            for key in flight_keys:
                flight_header[key.replace(' ', '_')] = one_flight['flight'][key]
            for tract in one_flight['track']:
                flight_tract = deepcopy(flight_header)
                for track_key in tract_keys:
                    flight_tract[track_key.replace(' ', '_')] = tract[track_key]
                flight_append(flight_tract)

    ''' Transform to dataframe '''
    col_order = [
        'Flight_ID',
        'Ident',
        'Origin',
        'Destination',
        'Actual_Arrival_Time_(UTC)',
        'DRemains',
        'TRemains',
        'TTravelled',
        'Time_(UTC)',
        'Latitude',
        'Longitude',
    ]
    pd.DataFrame(flights)[col_order].to_csv(
        flights_data_flag.value().replace('.json', '.csv'),
        index=False)
    print("Len:", len(flights))
    print(flights[0])

