#!/usr/bin/env sh

cdo -b F64 merge era5_10/*.nc era5_10.nc
cdo -b F64 merge era5_25/*.nc era5_25.nc
cdo -b F64 merge hres_10/*.grib hres_10.grib
cdo -b F64 merge hres_25/*.grib hres_25.grib