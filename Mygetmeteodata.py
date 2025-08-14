####Retrieve weather data for each location from KNMI GRIB file and save it as CSV

import os
import re
import numpy as np
import pandas as pd
from osgeo import gdal
from pcse.base import WeatherDataProvider, WeatherDataContainer

import warnings
warnings.filterwarnings("ignore", message="GRIB: .*", category=RuntimeWarning)

#function to compute daily weather data
def load_weather_data_knmi_complete(coord, folder_path="E:/dataset_month1",
                                    temp_band=26, dpt_band=28, rh_band=27,
                                    u_band=29, v_band=30, rain_band=24, rad_band=9,
                                    window_size=1):
    lat, lon = coord

    #
    grib_all = sorted([f for f in os.listdir(folder_path) if "HA43_N20" in f])

    # for rain and radiation
    grib_accum = [f for f in grib_all if f.endswith("00500_GB")]
    grib_all = [f for f in grib_all if not f.endswith("00500_GB") or f in grib_accum]

    daily_records = {}

    for fname in grib_all:
        fpath = os.path.join(folder_path, fname)
        dataset = gdal.Open(fpath)
        if dataset is None:
            continue

        gt = dataset.GetGeoTransform()
        origin_x, pixel_width, _, origin_y, _, pixel_height = gt
        x = int((lon - origin_x) / pixel_width)
        y = int((origin_y - lat) / abs(pixel_height))

        x_min = max(0, x - window_size)
        x_max = min(dataset.RasterXSize, x + window_size + 1)
        y_min = max(0, y - window_size)
        y_max = min(dataset.RasterYSize, y + window_size + 1)

        def read_patch(band_num):
            if band_num > dataset.RasterCount:
                return None
            band = dataset.GetRasterBand(band_num)
            arr = band.ReadAsArray().astype(np.float32)
            nodata = band.GetNoDataValue()
            if nodata is not None:
                arr[arr == nodata] = np.nan
            return arr[y_min:y_max, x_min:x_max]

        temp_patch = read_patch(temp_band)
        dpt_patch = read_patch(dpt_band)
        rh_patch = read_patch(rh_band)
        u_patch = read_patch(u_band)
        v_patch = read_patch(v_band)

        match = re.search(r"_([0-9]{8})[0-9]{4}_", fname)
        if not match:
            continue
        date_str = match.group(1)
        if date_str not in daily_records:
            daily_records[date_str] = {
                "temp": [], "vap": [], "wind": [], "rain": [], "rad": []
            }

        #actual vapor pressure
        def es_fao(T_C):
            return 0.610588 * np.exp((17.32491 * T_C) / (T_C + 238.102))

        if (temp_patch is not None and not np.all(np.isnan(temp_patch))
            and rh_patch is not None and not np.all(np.isnan(rh_patch))):
            es = es_fao(temp_patch)          # kPa
            vap = es * (rh_patch)    # kPa

            daily_records[date_str]["temp"].append(np.nanmean(temp_patch))
            daily_records[date_str]["vap"].append(np.nanmean(vap))
        else:
            if temp_patch is not None:
                daily_records[date_str]["temp"].append(np.nanmean(temp_patch))
                daily_records[date_str]["vap"].append(np.nan)


        # wind speed
        if u_patch is not None and v_patch is not None:
            wind = np.sqrt(u_patch**2 + v_patch**2)
            daily_records[date_str]["wind"].append(np.nanmean(wind))

        # rain and rad
        if fname.endswith("00500_GB"):
            rain_patch = read_patch(rain_band)
            rad_patch = read_patch(rad_band)

            if rain_patch is not None:
                daily_records[date_str]["rain"].append(np.nanmean(rain_patch))
            if rad_patch is not None:
                daily_records[date_str]["rad"].append(np.nanmean(rad_patch))

    # daily
    weather_dict = {}
    for date, v in daily_records.items():
        if not v["temp"] or not v["vap"] or not v["wind"]:
            continue

        tmin = float(np.nanmin(v["temp"]))
        tmax = float(np.nanmax(v["temp"]))
        vap = float(np.nanmean(v["vap"]))
        wind = float(np.nanmean(v["wind"]))
        rain = float(np.nansum(v["rain"])) if v["rain"] else 0.0
        rad = float(np.nansum(v["rad"])) if v["rad"] else 10000.0
        # irrad_kj = rad * 6 * 3600 / 1000  # W/m² → kJ/m²/day
        irrad_kj = float(np.nansum([np.nanmean(r) for r in v["rad"]])) / 1000  #  J/m² →  kJ/m²
     

        day = pd.to_datetime(date, format="%Y%m%d").date()
        weather_dict[day] = {
            "TMIN": tmin,
            "TMAX": tmax,
            "VAP": vap,
            "WIND": wind,
            "RAIN": rain,
            "IRRAD": irrad_kj
        }

    if not weather_dict:
        raise ValueError("please check the grib file or coordinates.")

    class MyWeatherProvider(WeatherDataProvider):
        def __init__(self, weatherdata, lat, lon, elev):
            super().__init__()
            self.weatherdata = weatherdata
            self._metadata = {"LAT": lat, "LON": lon, "ELEV": elev}

    return MyWeatherProvider(weather_dict, lat, lon, 1.0)

# function to save wather data as csv
def save_as_pcse_csv(df, output_csv_path, lat, lon, elev,
                     angstA=0.18, angstB=0.55, has_sunshine=False,
                     station="My Station", country="Netherlands", source="KNMI",
                     contact="Shijie", description="KNMI derived weather data"):
    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write("## Site Characteristics\n")
        f.write(f"Country = '{country}'; Station = '{station}'; Description = '{description}'; "
                f"Source = '{source}'; Contact = '{contact}'; "
                f"Longitude = {lon}; Latitude = {lat}; Elevation = {elev}; "
                f"AngstromA = {angstA}; AngstromB = {angstB}; HasSunshine = {str(has_sunshine)}\n")
        
        f.write("## Daily weather observations (missing values are NaN)\n")
        f.write("DAY,IRRAD,TMIN,TMAX,VAP,WIND,RAIN,SNOWDEPTH\n")
        
        for date, row in df.iterrows():
            day_str = date.strftime("%Y%m%d")
            irrad = round(row.get("IRRAD", float("nan")), 2)
            tmin = round(row.get("TMIN", float("nan")), 2)
            tmax = round(row.get("TMAX", float("nan")), 2)
            vap = round(row.get("VAP", float("nan")), 3)
            wind = round(row.get("WIND", float("nan")), 2)
            rain = round(row.get("RAIN", float("nan")), 2)
            # SNOWDEPTH 可以为 NaN
            f.write(f"{day_str},{irrad},{tmin},{tmax},{vap},{wind},{rain},NaN\n")

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def process_one_location(row, year, folder_path, save_dir):
    lat = row["latitude"]
    lon = row["longitude"]
    coord = (lat, lon)

    try:
        provider = load_weather_data_knmi_complete(coord, folder_path=folder_path)
        weather_dict = provider.weatherdata
        weather_df = pd.DataFrame.from_dict(weather_dict, orient="index")
        weather_df.index = pd.to_datetime(weather_df.index)

        filename = f"weather_{lat:.4f}_{lon:.4f}_{year}.csv"
        save_path = os.path.join(save_dir, filename)

        save_as_pcse_csv(
            weather_df,
            output_csv_path=save_path,
            lat=lat,
            lon=lon,
            elev=1.0
        )
        return f"Saved: {save_path}"
    
    except Exception as e:
        return f"Failed ({coord}): {e}"


def parallel_save_weather_csv(brp_df, year, folder_path, save_dir="./data/meteo/csv_barley_test", max_workers=4):
    os.makedirs(save_dir, exist_ok=True)
    brp_df = pd.read_csv(brp_csv_path)
    tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _, row in brp_df.iterrows():
            tasks.append(executor.submit(process_one_location, row, year, folder_path, save_dir))

        for future in as_completed(tasks):
            print(future.result())

if __name__ == "__main__":
    brp_csv_path = "C:/users/suliko/q34/25q3/thesis/fpcup_new/fpcup-main/barley_points.csv"
    folder_path = "E:/dataset_month1"
    parallel_save_weather_csv(brp_csv_path, year=2022, folder_path=folder_path, max_workers=6)
