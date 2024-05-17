"""Tools for checking quality and finding problems in the data."""

import warnings

import numpy as np
import pandas as pd
from annalist.annalist import Annalist

import hydrobot.utils as utils
from hydrobot.data_sources import QualityCodeEvaluator

annalizer = Annalist()


def gap_finder(data: pd.Series) -> list:
    """
    Find the indices and lengths of gaps (sequences of NaN values) in a pandas Series.

    Parameters
    ----------
    data : pd.Series
        Input Series containing NaN values.

    Returns
    -------
    list :
        List of tuples, each containing the index of a NaN value, the length of the gap
        containing it, and True for strictness.
    """
    # Find the indices where NaN values start and end
    idx0 = np.flatnonzero(np.r_[True, np.diff(pd.isna(data)) != 0, True])

    # Calculate the length of each gap
    count = np.diff(idx0)

    # Mask for the gaps that contain NaN values
    valid_mask = pd.isna(data.iloc[idx0[:-1]])

    # Select indices of gaps that contain NaN values
    out_idx = idx0[:-1][valid_mask]

    # Select lengths of gaps that contain NaN values
    out_count = count[valid_mask]

    # Select indices of NaN values in the original Series
    indices = data.iloc[out_idx].index

    # Create a list of tuples containing index, gap length, and strictness
    out = list(zip(indices, out_count, [True] * len(indices), strict=True))

    return out


def small_gap_closer(series: pd.Series, gap_limit: int) -> pd.Series:
    """
    Remove small gaps from a series.

    Gaps are defined by a sequential number of np.NaN values
    Small gaps are defined as gaps of length gap_length or less.

    Will return series with the nan values in the short gaps removed, and the
    long gaps untouched.

    Parameters
    ----------
    series : pandas.Series
        Data which has gaps to be closed
    gap_limit : integer
        Maximum length of gaps removed, will remove all np.NaN's in consecutive runs
        of gap_length or less

    Returns
    -------
    pandas.Series
        Data with any short gaps removed
    """
    gaps = gap_finder(series)
    for gap in gaps:
        if gap[1] <= gap_limit:
            # Determine the range of rows to remove
            mask = ~series.index.isin(
                series.index[
                    series.index.get_loc(gap[0]) : series.index.get_loc(gap[0]) + gap[1]
                ]
            )
            # Remove the bad rows
            series = pd.Series(series[mask])
    return series


def check_data_quality_code(
    series: pd.Series,
    check_series: pd.Series,
    qc_evaluator: QualityCodeEvaluator,
    gap_limit=10800,
) -> pd.DataFrame:
    """
    Quality Code Check Data.

    Quality codes data based on the difference between the standard series and
    the check data

    Parameters
    ----------
    series : pd.Series
        Data to be quality coded
    check_series : pd.Series
        Check data
    qc_evaluator : data_sources.QualityCodeEvaluator
        Handler for QC comparisons
    gap_limit : integer (seconds)
        If the nearest real data point is more than this many seconds away, return 200

    Returns
    -------
    pd.Series
        The QC values of the series, indexed by the END time of the QC period
    """
    first_data_date = series.index[0]
    last_data_date = series.index[-1]
    if check_series.empty and isinstance(first_data_date, pd.Timestamp):
        # Maybe you should go find that check data
        warnings.warn("Warning: No check data", stacklevel=2)
        return pd.DataFrame(
            columns=["Value", "Code", "Details"],
            index=[first_data_date],
        )
    first_check_date = check_series.index[0]
    last_check_date = check_series.index[-1]
    if (
        isinstance(first_data_date, pd.Timestamp)
        and isinstance(last_data_date, pd.Timestamp)
        and isinstance(first_check_date, pd.Timestamp)
        and isinstance(last_check_date, pd.Timestamp)
    ):
        # qc_series = pd.Series({first_data_date: np.NaN})
        qc_frame = pd.DataFrame(
            columns=["Value", "Code", "Details"],
            index=[first_data_date],
        )
        if first_check_date < first_data_date or last_check_date > last_data_date:
            # Can't check something that's not there
            raise KeyError(
                "Error: check data out of range. "
                f"First check date: {first_check_date}. "
                f"First data date: {first_data_date}. "
                f"Last check date: {last_check_date}. "
                f"Last data date: {last_data_date}. "
            )
        else:
            # Stuff actually working (hopefully)
            for check_time, check_value in check_series.items():
                if isinstance(check_time, pd.Timestamp):
                    adjusted_time = find_nearest_valid_time(series, check_time)
                    if abs((adjusted_time - check_time).total_seconds()) < gap_limit:
                        qc_value = qc_evaluator.find_qc(
                            series[adjusted_time], check_value
                        )
                    else:
                        qc_value = 200
                    qc_frame.loc[check_time, "Value"] = qc_value
                    qc_frame.loc[check_time, "Code"] = "CHK"
                    qc_frame.loc[check_time, "Details"] = (
                        f"Check value at {check_time} used to validate "
                        f"data value at {adjusted_time}."
                    )
                else:
                    raise KeyError("Series indices should be pandas.Timestamp.")
            qc_frame = qc_frame.shift(periods=-1)
            qc_frame.loc[qc_frame.index[-1], "Value"] = 0
        return qc_frame
    else:
        raise KeyError("Series indices should be pandas.Timestamp.")


def missing_data_quality_code(std_series, qc_data, gap_limit):
    """
    Make sure that missing data is QC100.

    Returns qc_frame with QC100 values added where std_series is NaN

    Parameters
    ----------
    std_series : pd.Series
        Base series which may contain NaNs
    qc_data
        QC series for base std_series without QC100 values
    gap_limit
        Maximum size of gaps which will be ignored

    Returns
    -------
    pd.Series
        The modified QC series, indexed by the start time of the QC period
    """
    for gap in gap_finder(std_series):
        if gap[1] > gap_limit:
            end_idx = std_series.index.get_loc(gap[0]) + gap[1]
            # end of gap should recover the value from previous
            if end_idx < len(std_series):
                prev_qc_data = qc_data[qc_data.index <= std_series.index[end_idx]]
                prev_qc_data = prev_qc_data[prev_qc_data["Value"] > 100]
                prev_qc_data = prev_qc_data.sort_index()
                qc_data.loc[std_series.index[end_idx]] = prev_qc_data.iloc[-1]
                if std_series.index[end_idx] in prev_qc_data.index:
                    qc_data.loc[std_series.index[end_idx], "Details"] = (
                        qc_data.loc[std_series.index[end_idx], "Details"]
                        + f" [End of gap which started at {gap[0]}]"
                    )
                else:
                    qc_data.loc[std_series.index[end_idx], "Details"] = (
                        f"End of gap which started at {gap[0]}. "
                        f"Returning to QC code first assigned at {prev_qc_data.index[-1]}"
                    )
                qc_data = qc_data.sort_index()

            # getting rid of any stray QC codes in the middle
            drop_series = qc_data["Value"]
            drop_series = drop_series[drop_series.index > gap[0]]
            drop_series = drop_series[
                drop_series.index <= std_series.index[end_idx - 1]
            ]
            qc_data = qc_data.drop(drop_series.index)

            # start of gap
            qc_data.loc[gap[0], "Value"] = 100
            qc_data.loc[gap[0], "Code"] = "GAP"
            if end_idx >= len(std_series):
                gap_end = std_series.index[-1]
            else:
                gap_end = std_series.index[end_idx]
            qc_data.loc[
                gap[0], "Details"
            ] = f"Missing data amounting to {(gap_end - gap[0])}"
            qc_data = qc_data.sort_index()

    return qc_data.sort_index()


def find_nearest_time(series, dt):
    """
    Find the time in the series that is closest to dt.

    For example for the series::

        pd.Timestamp("2021-01-01 02:00"): 0.0,
        pd.Timestamp("2021-01-01 02:15"): 0.0,

    with dt::

        pd.Timestamp("2021-01-01 02:13"): 0.0,

    the result should be the closer ``pd.Timestamp("2021-01-01 02:15")`` value

    Parameters
    ----------
    series : pd.Series
        The series indexed by time

    dt : Datetime
        Time that may or may nor exactly line up with the series

    Returns
    -------
    Datetime
        The value of dt rounded to the nearest timestamp of the series

    """
    # Make sure it is in the range

    first_timestamp = series.index[0]
    last_timestamp = series.index[-1]
    if dt < first_timestamp or dt > last_timestamp:
        raise KeyError("Timestamp not within data range")

    output_index = series.index.get_indexer([dt], method="nearest")
    return series.index[output_index][0]


def find_nearest_valid_time(series, dt) -> pd.Timestamp:
    """
    Find the time in the series that is closest to dt, but ignoring NaN values (gaps).

    Parameters
    ----------
    series : pd.Series
        The series indexed by time
    dt : Datetime
        Time that may or may nor exactly line up with the series

    Returns
    -------
    Datetime
        The value of dt rounded to the nearest timestamp of the series

    """
    # Make sure it is in the range
    first_timestamp = series.index[0]
    last_timestamp = series.index[-1]
    if dt < first_timestamp or dt > last_timestamp:
        raise KeyError("Timestamp not within data range")

    series = series.dropna()
    output_index = series.index.get_indexer([dt], method="nearest")
    return series.index[output_index][0]


def base_data_qc_filter(std_series, qc_filter):
    """
    Filter out data based on quality code filter.

    Return only the base series data for which the next date in the qc_filter
    is 'true'

    Parameters
    ----------
    std_series : pandas.Series
        Data to be filtered
    qc_filter : pandas.Series of booleans
        Dates for which some condition is met or not

    Returns
    -------
    pandas.Series
        The filtered data

    """
    base_filter = qc_filter.reindex(std_series.index, method="ffill").fillna(False)
    return std_series[base_filter]


def base_data_meets_qc(std_series, qc_series, target_qc):
    """
    Find all data where QC targets are met.

    Returns only the base series data for which the next date in the qc_filter is
    equal to target_qc

    Parameters
    ----------
    std_series: pandas.Series
        Data to be filtered
    qc_series: pandas.Series
        quality code data series, some of which are presumably target_qc
    target_qc: int
        target quality code

    Returns
    -------
    pandas.Series
        Filtered data
    """
    return base_data_qc_filter(std_series, qc_series == target_qc)


def diagnose_data(std_series, check_series, qc_series, frequency):
    """
    Return description of how much missing data, how much for each QC, etc.

    This function feels like a mess, I'm sorry.
    The good news is that it is only a diagnostic, so feel free to change the hell
    out of it

    Parameters
    ----------
    std_series : pandas.Series
        processed base time series data
    check_series : pandas.Series
        Check datatime series
    qc_series : pandas.Series
        QC time series
    frequency : DateOffset or str
        Frequency to which the data gets set to

    Returns
    -------
    None
        Prints statements that describe the state of the data
    """
    # total time
    first_timestamp = std_series.index[0]
    last_timestamp = std_series.index[-1]
    total_time = last_timestamp - first_timestamp
    print(f"Time examined is {total_time} from {first_timestamp} to {last_timestamp}")
    print(
        f"Have check data for {check_series.index[-1] - first_timestamp} "
        f"(last check {check_series.index[-1]})"
    )

    # periods
    ave_period = pd.to_timedelta(frequency)  # total_time / (len(raw_data) - 1)
    gap_time = ave_period * (len(std_series) - len(std_series.dropna()) + 1)
    print(f"Missing {gap_time} of data, that's {gap_time/total_time*100}%")

    # QCs
    split_data = splitter(std_series, qc_series, frequency)
    for qc in split_data:
        print(
            f"Data that is QC{qc} makes up "
            f"{len(split_data[qc].dropna()) / len(std_series.dropna()) * 100:.2f}% "
            "of the workable data and "
            f"{len(split_data[qc].dropna()) / len(std_series) * 100:.2f}% "
            "of the time period"
        )


def splitter(std_series, qc_series, frequency):
    """
    Split the data up by QC code.

    Selects all data which meets a given QC code, pads the rest with NaN values
    Does this for all current NEMs values ([0, 100, 200, 300, 400, 500, 600])

    Parameters
    ----------
    std_series : pd.Series
        Time series data to be split up
    qc_series : pd.Series
        QC values to split the data by
    frequency : DateOffset or str
        Frequency to which the data gets set to

    Returns
    -------
    dict of int:pd.Series pairs
        Keys are the QC values as ints, values are series of data that fits
    """
    qc_list = [0, 100, 200, 300, 400, 500, 600]
    return_dict = {}

    for qc in qc_list:
        if qc == 100:
            return_dict[qc] = (
                base_data_meets_qc(std_series, qc_series, qc)
                .fillna(std_series.median())
                .asfreq(frequency)
            )
        else:
            return_dict[qc] = base_data_meets_qc(std_series, qc_series, qc).asfreq(
                frequency
            )

    return return_dict


def max_qc_limiter(qc_frame: pd.DataFrame, max_qc) -> pd.DataFrame:
    """
    Enforce max_qc on a QC series.

    Replaces all values with QCs above max_qc with max_qc

    Parameters
    ----------
    qc_frame : pd.DataFrame
        The series to be limited.
    max_qc : numerical
        maximum allowed value. None imposes no limit.

    Returns
    -------
    pd.DataFrame
        qc_frame with too high QCs limited to max_qc
    """
    clipped_data = qc_frame["Value"].clip(np.NaN, max_qc)

    diff_idxs = qc_frame[qc_frame["Value"] != clipped_data].index

    qc_frame.loc[diff_idxs, "Code"] = qc_frame.loc[diff_idxs, "Code"] + ", LIM"
    qc_frame.loc[diff_idxs, "Details"] = (
        qc_frame.loc[diff_idxs, "Details"]
        + f" [Site QC limit applies to a maximum of {max_qc}.]"
    )
    qc_frame["Value"] = clipped_data

    return qc_frame


def quality_encoder(
    base_series: pd.Series,
    check_series: pd.Series,
    qc_evaluator: QualityCodeEvaluator,
    gap_limit: int,
    max_qc=np.NaN,
    interval_dict: dict = None,
) -> pd.Series:
    """
    Return complete QC series.

    Parameters
    ----------
    base_series : pd.Series
        Base time series data
    check_series : pd.Series
        Check data time series
    qc_evaluator : data_sources.QualityCodeEvaluator
        Handler for QC comparisons
    gap_limit : int
        Maximum size of gaps which will be ignored
    max_qc : numeric
        Maximum allowed QC value
    interval_dict : dict
        Key:Value pairs of max_interval:downgraded_qc for single_downgrade_out_of_validation

    Returns
    -------
    pd.Series
        The modified QC series, indexed by the start time of the QC period
    """
    qc_series = check_data_quality_code(base_series, check_series, qc_evaluator)
    if interval_dict is None:
        interval_dict = {}
    qc_series = bulk_downgrade_out_of_validation(
        qc_series, check_series, interval_dict=interval_dict
    )
    qc_series = missing_data_quality_code(base_series, qc_series, gap_limit=gap_limit)

    qc_series = max_qc_limiter(qc_series, max_qc)
    # qc_series.index.name = "Time"
    # qc_series.name = "Value"
    return qc_series


def bulk_downgrade_out_of_validation(
    qc_frame: pd.DataFrame,
    check_series: pd.Series,
    interval_dict: dict,
    day_end_rounding: bool = True,
):
    """
    Applies caps on quality codes for any data that has gaps between check data that is too large.

    Utilises single_downgrade_out_of_validation multiple times for different time periods.

    Parameters
    ----------
    qc_frame : pd.DataFrame
        Quality series that potentially needs downgrading
    check_series : pd.Series
        Check series to check for frequency of checks
    interval_dict : dict
        Key:Value pairs of max_interval:downgraded_qc for single_downgrade_out_of_validation
    day_end_rounding : bool
        Whether to round to the day end. If true, downgraded data starts at midnight

    Returns
    -------
    pd.DataFrame
        The qc_frame with any downgraded QCs added in

    """
    if not qc_frame.empty and not check_series.empty:
        for key in interval_dict:
            qc_frame = single_downgrade_out_of_validation(
                qc_frame, check_series, key, interval_dict[key], day_end_rounding
            )
    return qc_frame


def single_downgrade_out_of_validation(
    qc_frame: pd.DataFrame,
    check_series: pd.Series,
    max_interval: pd.DateOffset,
    downgraded_qc: int = 200,
    day_end_rounding: bool = True,
):
    """
    Applies a cap on quality codes for any data that has gaps between check data that is too large.

    Only applies a single cap quality code, see bulk_downgrade_out_of_validation for multiple steps.

    Parameters
    ----------
    qc_frame : pd.DataFrame
        Quality series that potentially needs downgrading
    check_series : pd.Series
        Check series to check for frequency of checks
    max_interval : pd.DateOffset
        How long of a gap between checks before the data gets downgraded
    downgraded_qc : int
        Which code the quality data gets downgraded to
    day_end_rounding : bool
        Whether to round to the day end. If true, downgraded data starts at midnight

    Returns
    -------
    pd.DataFrame
        The qc_frame with any downgraded QCs added in
    """
    if qc_frame.empty or check_series.empty:
        raise ValueError(
            "Cannot have empty qc series or check series. qc.empty = {}, check.empty = {}".format(
                qc_frame.empty, check_series.empty
            )
        )
    # When they should have their next check by
    due_date = check_series.index + max_interval
    due_date = due_date[:-1]
    if day_end_rounding:
        due_date = due_date.ceil("D")
    # Whether there has been a check since then
    overdue = (due_date < check_series.index[1:]) & (
        qc_frame.loc[check_series.index[:-1], "Value"] > downgraded_qc
    )
    # Select overdue times
    unvalidated = due_date[overdue]
    downgraded_times = pd.DataFrame(
        {
            "Value": [downgraded_qc for _ in unvalidated],
            "Code": ["OOV" for _ in unvalidated],
            "Details": [
                "Site inspection overdue. Last inspection at "
                f"{check_series.index[idx]}. Data downgraded to QC{downgraded_qc} "
                "until next inspection."
                for idx in range(len(unvalidated))
            ],
        },
        index=unvalidated,
    )

    # combine and sort
    if not downgraded_times.empty:
        qc_frame = pd.concat([qc_frame, downgraded_times]).sort_index()
    # qc_frame.loc[qc_frame.index[-1], "Value"] = 0

    return qc_frame


def cap_qc_where_std_high(std_frame, qc_frame, cap_qc, cap_threshold):
    """
    Cap the quality code of data where the standard series exceeds some value.

    Parameters
    ----------
    std_frame : pd.DataFrame
    qc_frame : pd.DataFrame
    cap_qc : numeric
    cap_threshold : numeric


    Returns
    -------
    pd.DataFrame
        the qc series to return
    """
    if qc_frame.empty or std_frame.empty:
        raise ValueError("qc series can't be empty for function cap_qc_where_std_high")
    std_series = std_frame["Value"]
    capped_data = std_series > cap_threshold
    capped_qc_changes = capped_data.loc[capped_data.shift() != capped_data]  # noqa
    potential_new_qc = capped_qc_changes.replace(True, cap_qc).replace(False, np.NaN)
    new_qc = utils.compare_two_qc_take_min(potential_new_qc, qc_frame["Value"])

    qc_frame = qc_frame.reindex(new_qc.index, method="ffill")

    diff_idxs = qc_frame[qc_frame["Value"] != new_qc].index

    qc_frame.loc[diff_idxs, "Code"] = qc_frame.loc[diff_idxs, "Code"] + ", CAP"
    qc_frame.loc[diff_idxs, "Details"] = (
        qc_frame.loc[diff_idxs, "Details"]
        + f" [DO above {cap_qc} means apply a maximum qc of {cap_threshold}.]"
    )
    qc_frame["Value"] = new_qc

    return qc_frame
