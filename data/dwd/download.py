import glob
import os
import re
import itertools

import codecs
from io import StringIO

import ftplib
import logging
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import date_range
from pytest import param
import xarray as xr

from tqdm import tqdm


ftp_server = 'opendata.dwd.de'


def download(data_list, logger):
    """ Download weather stations information and weather station data as txt files. """
    ftp_con = ftplib.FTP(ftp_server, 'anonymous', 'anonymous')
    for ftp_dir, stations_file, output_dir, _ in data_list:    
        # try to download stations file
        logger.info(f'Downloading stations file {stations_file} to {output_dir}.txt.')
        try:
            ftp_download_file(ftp_con, ftp_dir, stations_file, f'{output_dir}.txt')
        except:
            logger.error('Stations file not found! Skipping...', exc_info=True)
            continue

        # try to parse stations file
        logger.info(f'Parse {output_dir}.txt and load pandas DataFrame.')
        try:
            stations_frame = parse_weather_stations_file(f'{output_dir}.txt')
        except:
            logger.error(f'Error while parsing {output_dir}.txt! Skipping...', exc_info=True)
            continue

        # try to download weather station data
        logger.info(f'Downloading weather station data to {output_dir}.')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        try:
            for id in tqdm(stations_frame.index):
                logging.info(f'Downloading weather data for ID {id:05}.')
                files = ftp_list_files(ftp_con, ftp_dir, filter_regex=f'.*_{id:05}_.*')
                if len(files) == 0:
                    logger.warning(f'No weather station data found for {id:05}. Skipping...')
                    continue
                if len(files) > 1:
                    logger.warning(f'Found multiple weather station files for {id:05}. Downloading only first one...')
                    logger.warning(files)
                    files = files[:1]
                ftp_download_file(ftp_con, ftp_dir, files[0], 'temp.zip')
                weather_data_path = extract_zip(output_dir)
                os.remove('temp.zip')
        except:
            logger.error('Error while downloading weather station!', exc_info=True)


def get_timezone(weather_station_id, output_dir):
    """ Read parameter csv and return timezone for 2015 and later. """
    parameter_file = glob.glob(os.path.join(output_dir, f'*Parameter*_{weather_station_id:05}.txt'))[0]
    with open(parameter_file, 'r', encoding='ISO-8859-1') as csv_file:
        csv_lines = csv_file.readlines()
    csv = ''.join([line for line in csv_lines if 'eor;' in line])
    parameter_df = pd.read_csv(StringIO(csv), sep=';')
    parameter_df['start_year'] = pd.to_datetime(parameter_df['Von_Datum'], format='%Y%m%d').dt.year
    parameter_df['end_year'] = pd.to_datetime(parameter_df['Bis_Datum'], format='%Y%m%d').dt.year
    selections = (parameter_df['start_year'] >= 2015) | (parameter_df['end_year'] >= 2015)
    if selections.sum() > 0:
        tz_columns = parameter_df.loc[selections]['Zusatz-Info']
        tz = np.unique(tz_columns)
        if len(tz) > 1:
            raise ValueError(f'No definite timezone for weather station {weather_station_id}.')
        else:
            tz = tz[0].lower()
            if 'utc' in tz:
                return 'utc'
            elif 'mez' in tz:
                return 'mez'


def get_max_length_missing_values(weather_frame):
    nans = weather_frame.isna().values
    occurences = [(k, len(list(g)))
                  for k, g in itertools.groupby(nans)
                  if k == True]
    if len(occurences) == 0:
        return 0
    else:
        lengths = np.array([num for _, num in occurences])
        return lengths.max()


def create_dataset(data_list, logger, date_range):
    """ Read all weather station txt files and combine it into a single dataset. """
    ds_dict = {}
    for _, _, output_dir, columns in data_list:
        # try to parse stations file
        logger.info(f'Parse {output_dir}.txt and load pandas DataFrame.')
        try:
            stations_frame = parse_weather_stations_file(f'{output_dir}.txt')
            stations_frame['N/A'] = False
        except:
            logger.error(f'Error while parsing {output_dir}.txt! Skipping...', exc_info=True)
            continue

        # filter for relevant weather stations
        start_date = date_range[0]
        end_date = date_range[-1]
        stations_frame = stations_frame.loc[
            np.logical_and(
                (stations_frame['von_datum'] < start_date),
                (stations_frame['bis_datum'] > end_date)
            )
        ]

        # prepare variables for collecting data
        weather_station_ids = list()
        weather_station_lats = list()
        weather_station_longs = list()
        weather_data = dict()
        for _, y in columns.items():
            weather_data[y] = list()

        # iterate over weather staions and collect data
        for weather_station_idx in tqdm(stations_frame.index):
            weather_station = stations_frame.loc[weather_station_idx]
            weather_station_id = weather_station.name

            matching_data_files = glob.glob(os.path.join(output_dir, f'produkt*_{weather_station_id:05}.txt'))
            if len(matching_data_files) > 1:
                logger.warning(f'Found multiple data files for weather station {weather_station_id}.')
            elif len(matching_data_files) == 0:
                logger.info(f'No data file found for weather station {weather_station_id}.')
            else:
                # get timezone and transform to utc
                weather_frame = parse_weather_station_data(matching_data_files[0])
                timezone = get_timezone(weather_station_id, output_dir)
                if timezone == 'utc':
                    pass
                elif timezone == 'mez':
                    weather_frame.index = weather_frame.index - pd.to_timedelta('1h')
                else:
                    raise NotImplemented(f'Not implemented timezone for weather station {weather_station_id}.')

                # interpolate missing data if possible
                weather_frame = weather_frame[list(columns.keys())]
                weather_frame = weather_frame.reindex(date_range, fill_value=np.nan)
                for col in weather_frame.columns:
                    weather_frame[col].replace(-999, np.nan, inplace=True)
                max_length_missing_values = get_max_length_missing_values(weather_frame)
                if max_length_missing_values <= 24:
                    weather_frame = weather_frame.to_xarray() \
                                                 .dropna('index') \
                                                 .interp(index=date_range, method='linear',
                                                         kwargs={"fill_value": "extrapolate"}) \
                                                 .to_pandas()
                else:
                    logger.warning(f'Skipping {weather_station_id} due to {max_length_missing_values} '
                                    'maximum consequtive missing values in selected date range.')
                    continue

                weather_station_ids.append(weather_station_id)
                weather_station_lats.append(weather_station.geoBreite)
                weather_station_longs.append(weather_station.geoLaenge)
                for key, value in columns.items():
                    weather_data[value].append(weather_frame[key].values)

        # create xarray dataset
        data_vars = dict(
            latitude=(['station_id'], weather_station_lats),
            longitude=(['station_id'], weather_station_longs)
        )
        for key, values in weather_data.items():
            data_vars[key] = (
                ['station_id', 'time'],
                np.array(values)
            )
        ds_dict[output_dir] = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                station_id=(['station_id'], weather_station_ids),
                time=(['time'], date_range)
            )
        )

    return ds_dict


def ftp_download_file(ftp_con, dir, file, output):
    """ Download stations txt file given the file name and ftp directory path. """
    files = ftp_list_files(ftp_con, dir)
    if file in files:
        # save stations file if exists
        ftp_con.cwd(dir)
        ftp_con.retrbinary("RETR " + file, open(output, 'wb').write)
    else:
        # no stations file found
        raise Exception('File not found!')


def ftp_list_files(ftp_con, dir, filter_regex=None):
    """ List all files of a directory in a ftp connection """
    files = []
    ftp_con.cwd(dir)
    ftp_con.dir(files.append)
    files = np.vectorize(lambda file: re.sub(' +', ' ', file))(files)
    files = np.vectorize(lambda file: ' '.join(file.split(' ')[8:]))(files)

    if filter_regex is not None:
        filter = np.vectorize(lambda file: re.match(filter_regex, file) is not None)(files)
        files = files[filter]

    return files


def extract_zip(output_dir, zip_path='temp.zip'):
    """ Extract the weather data from zip file. """
    with ZipFile(zip_path) as zip:
        files = [f for f in zip.namelist() if 'produkt' in f]
        zip.extract(files[0], path=output_dir)
        files = [f for f in zip.namelist() if 'Parameter' in f and '.txt' in f]
        zip.extract(files[0], path=output_dir)
    return os.path.join(output_dir, files[0])


def parse_weather_stations_file(path, time_indices=['von_datum', 'bis_datum']):
    """ Parse the weather stations file containing meta information of all weather stations. """
    # open file and read lines
    with codecs.open(path, 'r', encoding='iso-8859-1', errors='ignore') as f:
        lines = f.readlines()

    # define a parse function that will be applied to each line
    # to match the comma separated value format (CSV)
    def parse_line(line):
        while '  ' in line:
            line = line.replace('  ', ' ')
        line = line.replace(' \r\n', '')
        line = line.split(' ')
        if len(line) > 8:
            # assume only cities contain spaces
            city = ' '.join(line[6:-1])
            line = line[:6] + [city] + [line[-1]]
        line = ';'.join(line)
        return line
    lines = np.vectorize(parse_line)(lines)

    # skip second line because it contains only '-----'
    # and load pandas DataFrame
    header = lines[0]
    lines = lines[2:]
    csv_string = header + '\n'.join(lines)
    csv_file = StringIO(csv_string)
    frame = pd.read_csv(csv_file, sep=';')
    frame.set_index('Stations_id', inplace=True)

    # parse dates
    for time_col in time_indices:
        frame[time_col] = pd.to_datetime(frame[time_col], format='%Y%m%d')

    return frame


def parse_weather_station_data(file, time_index='MESS_DATUM'):
    """ Parse weather station data file containing weather information """
    # load raw string and remove unnecessary spaces
    # NOTE: Assume theses files do not contain spaces that are needed
    with open(file, 'r') as f:
        raw = f.read()
    raw = re.sub(' +', '', raw)

    # load pandas data frame and remove first (station id) and last columns (eor)
    frame = pd.read_csv(StringIO(raw), sep=';')
    frame = frame.iloc[:, 1:-1]

    # parse dates
    frame[time_index] = np.vectorize(lambda value: re.sub(':.*', '', value))(frame[time_index].astype(str))
    frame[time_index] = pd.to_datetime(frame[time_index], format='%Y%m%d%H')

    # remove duplicates and set time column as index
    frame.drop_duplicates(subset=time_index, keep='last', inplace=True)
    frame.set_index(time_index, inplace=True)

    return frame


if __name__ == '__main__':
    data_list = [
        (
            '/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/',
            'TU_Stundenwerte_Beschreibung_Stationen.txt',
            'temperature',
            {
                'TT_TU': 't2m'
            }
        ),
        (
            '/climate_environment/CDC/observations_germany/climate/hourly/solar/',
            'ST_Stundenwerte_Beschreibung_Stationen.txt',
            'solar',
            {
                'FG_LBERG': 'ssr',
            }
        ),
        (
            '/climate_environment/CDC/observations_germany/climate/hourly/wind/historical/',
            'FF_Stundenwerte_Beschreibung_Stationen.txt',
            'wind',
            {
                'F': 'u+v',
            }
        )
    ]

    logging.basicConfig(format='[%(levelname)s][%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger()

    # download(data_list, logger)
    ds_dict = create_dataset(data_list, logger, date_range=pd.date_range('2015-01-01 00:00', '2019-12-31 23:00', freq='1h'))

    for key in ds_dict:
        ds_dict[key].to_netcdf(f'{key}.nc')
