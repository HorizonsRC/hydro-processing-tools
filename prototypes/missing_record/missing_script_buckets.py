"""Missing record script."""
import csv
import time
import warnings

import numpy as np
import pandas as pd
import site_list_merge
import yaml
from pandas.tseries.frequencies import to_offset

from hydrobot.data_acquisition import get_data
from hydrobot.utils import infer_frequency

warnings.filterwarnings("ignore", message=".*Empty hilltop response:.*")

with open("script_config.yaml") as file:
    config = yaml.safe_load(file)

sites = site_list_merge.get_sites()
# sites = pd.DataFrame(
#     {
#         "SiteName": ["Lake William", "Manawatu at Teachers College"],
#         "RegionName": ["LAKES AND WQ", "Central"],
#     }
# )

with open("Active_Measurements_buckets.csv", newline="") as f:
    reader = csv.reader(f)
    measurements = [(row[0], row[1]) for row in reader]

region_stats_dict = {
    "Northern": [],
    "Eastern": [],
    "Central": [],
    "Special_P": [],
}
regions_list = {
    "Northern": ["NORTHERN"],
    "Eastern": ["EASTERN"],
    "Central": ["CENTRAL"],
    "Special_P": ["LAKES AND WQ", "Arawhata Piezometers"],
}
for _, site in sites.iterrows():
    for region in regions_list:
        if site.RegionName in regions_list[region]:
            region_stats_dict[region].append(site.SiteName)


def report_missing_record(site, measurement, start, end):
    """Reports minutes missing."""
    _, blob = get_data(
        config["base_url"],
        config["hts"],
        site,
        measurement,
        start,
        end,
    )

    if blob is None or len(blob) == 0:
        return np.nan

    series = blob[0].data.timeseries[blob[0].data.timeseries.columns[0]]
    series.index = pd.DatetimeIndex(series.index)

    freq = infer_frequency(series.index, method="mode")
    series = series.reindex(pd.date_range(start, end, freq=freq))
    missing_points = series.asfreq(freq).isna().sum()
    return str(missing_points * pd.to_timedelta(to_offset(freq)))


all_stats_dict = {}
start_timer = time.time()
for _, site in sites.iterrows():
    site_stats_list = []
    for meas in measurements:
        try:
            site_stats_list.append(
                report_missing_record(
                    site["SiteName"], meas[0], config["start"], config["end"]
                )
            )
        except ValueError as e:
            print(f"Site '{site['SiteName']}' with meas '{meas[0]}' doesn't work: {e}")
            site_stats_list.append(np.nan)

    all_stats_dict[site["SiteName"]] = site_stats_list
    print(site.SiteName, time.time() - start_timer)

# convert to measurement bucket
measurement_buckets = list(set([m[1] for m in measurements]))

bucket_stats_dict = {}
for site in all_stats_dict:
    site_bucket_dict = dict([(m, []) for m in measurement_buckets])
    for i in zip(measurements, all_stats_dict[site], strict=True):
        site_bucket_dict[i[0][1]].append(i[1])
    for bucket in site_bucket_dict:
        nanless = [m for m in site_bucket_dict[bucket] if m is not np.nan]
        if len(nanless) == 0:
            site_bucket_dict[bucket] = np.nan
        elif len(nanless) == 1:
            site_bucket_dict[bucket] = nanless[0]
        else:
            print("max used")
            print(site, bucket, nanless)
            # sum
            # site_bucket_dict[bucket] = sum([pd.to_timedelta(n) for n in nanless], pd.to_timedelta('0'))
            site_bucket_dict[bucket] = max([pd.to_timedelta(n) for n in nanless])

    bucket_stats_dict[site] = [site_bucket_dict[m] for m in measurement_buckets]

with open("output_dump/output.csv", "w", newline="") as output:
    wr = csv.writer(output)
    wr.writerow(["Sites"] + measurement_buckets)
    for site in bucket_stats_dict:
        wr.writerow([site] + bucket_stats_dict[site])

diff = pd.to_datetime(config["end"]) - pd.to_datetime(config["start"])

with open("output_dump/output_percent.csv", "w", newline="") as output:
    wr = csv.writer(output)
    wr.writerow(["Sites"] + measurement_buckets)
    for site in bucket_stats_dict:
        wr.writerow(
            [site]
            + [
                (i / diff) * 100 if i is not np.NaN else np.NaN
                for i in bucket_stats_dict[site]
            ]
        )

for region in regions_list:
    with open(f"output_dump/output_{region}.csv", "w", newline="") as output:
        wr = csv.writer(output)
        wr.writerow(["Sites"] + measurement_buckets)
        for site in region_stats_dict[region]:
            wr.writerow(
                [site]
                + [
                    (i / diff) * 100 if i is not np.NaN else np.NaN
                    for i in bucket_stats_dict[site]
                ]
            )
