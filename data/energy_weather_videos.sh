#!/usr/bin/env sh
ffmpeg -i ecmwf/era5_10_t2m.mp4 -i opsd/opsd_load.mp4 -filter_complex vstack=inputs=2 energy_weather_load_vertical.mp4
ffmpeg -i ecmwf/era5_10_ssr.mp4 -i opsd/opsd_solar.mp4 -filter_complex vstack=inputs=2 energy_weather_solar_vertical.mp4
ffmpeg -i ecmwf/era5_10_u+v.mp4 -i opsd/opsd_wind.mp4 -filter_complex vstack=inputs=2 energy_weather_wind_vertical.mp4

ffmpeg -i ecmwf/era5_10_t2m.mp4 -i opsd/opsd_load.mp4 -filter_complex hstack=inputs=2 energy_weather_load_horizontal.mp4
ffmpeg -i ecmwf/era5_10_ssr.mp4 -i opsd/opsd_solar.mp4 -filter_complex hstack=inputs=2 energy_weather_solar_horizontal.mp4
ffmpeg -i ecmwf/era5_10_u+v.mp4 -i opsd/opsd_wind.mp4 -filter_complex hstack=inputs=2 energy_weather_wind_horizontal.mp4
