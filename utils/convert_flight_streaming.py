from copy import deepcopy
import csv
import json

from atc import flags
from atc.utils.progress_bar_utils import print_progress_bar

flights_data_flag = flags.create(
    'flights_data',
    flags.FlagType.STRING,
    "Full path to the trajectory file in json format",
    required=True)


def get_num_lines_in_file(file_path):
    from subprocess import check_output
    return int(check_output(
        ['wc', '-l', file_path]).split(b' ')[0])


if __name__ == '__main__':
    flags.parse_flags()
    print("Filepath :", flights_data_flag.value())
    flights = []
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
    col_order = [
        'Flight_ID',
        'Ident',
        'Origin',
        'Destination',
        'OD_pair',
        'Actual_Arrival_Time_(UTC)',
        'DRemains',
        'TRemains',
        'TTravelled',
        'Time_(UTC)',
        'Latitude',
        'Longitude',
    ]
    fin = open(flights_data_flag.value().replace('.json', '.csv'), 'w')
    writer = csv.DictWriter(fin, fieldnames=col_order)
    writer.writeheader()
    num_lines = get_num_lines_in_file(flights_data_flag.value())
    flights_id = []
    with open(flights_data_flag.value()) as fin:
        for i, line in enumerate(fin):
            print_progress_bar(i+1, num_lines)
            one_flight = (json.loads(line))
            flight_header = {}
            if one_flight['flight']['Flight ID'] in flights_id:
                print("FlightID overlap: %s" % one_flight['flight']['Flight ID'])
            flights_id.append(one_flight['flight']['Flight ID'])
            for key in flight_keys:
                flight_header[key.replace(' ', '_')] = one_flight['flight'][key]
            for tract in one_flight['track']:
                flight_tract = deepcopy(flight_header)
                for track_key in tract_keys:
                    flight_tract[track_key.replace(' ', '_')] = tract[track_key]
                flight_tract['OD_pair'] = flight_tract['Origin'] + '-' + flight_tract['Destination']
                writer.writerow(flight_tract)
    fin.close()

