"""General filtering utilities."""


import numpy as np
import pandas as pd
from annalist.annalist import Annalist

annalizer = Annalist()


def clip(unclipped: pd.Series, low_clip: float, high_clip: float):
    """Clip values in a pandas Series within a specified range.

    Parameters
    ----------
    unclipped : pandas.Series
        Input data to be clipped.
    high_clip : float
        Upper bound for clipping. Values greater than this will be set to NaN.
    low_clip : float
        Lower bound for clipping. Values less than this will be set to NaN.

    Returns
    -------
    pandas.Series
        A Series containing the clipped values with the same index as the
        input Series.
    """
    unclipped_arr = unclipped.to_numpy()

    # Create a boolean condition for values that need to be clipped
    clip_cond = (unclipped_arr > high_clip) | (unclipped_arr < low_clip)

    # Use pandas' where function to clip values to NaN where the condition is
    # True
    clipped_series = unclipped.where(~clip_cond, np.nan)

    return clipped_series


# noinspection SpellCheckingInspection
def fbewma(input_data, span: int):
    """Calculate the Forward-Backward Exponentially Weighted Moving Average (FBEWMA).

    Parameters
    ----------
    input_data : pandas.Series
        Input time series data to calculate the FBEWMA on.
    span : int
        Span parameter for exponential weighting.

    Returns
    -------
    pandas.Series
        A Series containing the FBEWMA values with the same index as the
        input Series.
    """
    # Calculate the Forward EWMA.
    fwd = input_data.ewm(span=span).mean()

    # Calculate the Backward EWMA. (x[::-1] is the reverse of x)
    bwd = input_data[::-1].ewm(span=span).mean()

    # Stack fwd and the reverse of bwd on top of each other.
    stacked_ewma = pd.concat([fwd, bwd[::-1]])

    # Calculate the FB-EWMA by taking the mean between fwd and bwd.
    return stacked_ewma.groupby(level=0).mean()


def remove_outliers(input_data: pd.Series, span: int, delta: float):
    """Remove outliers.

    Remove outliers from a time series by comparing it to the
    Forward-Backward Exponentially Weighted Moving Average (FBEWMA).

    Parameters
    ----------
    input_data : pandas.Series
        Input time series data.
    span : int
        Span parameter for exponential weighting used in the FBEWMA.
    delta : float
        Threshold for identifying outliers. Values greater than this
        threshold will be set to NaN.

    Returns
    -------
    pandas.Series
        A Series containing the time series with outliers removed with
        the same index as the input Series.
    """
    # Calculate the FBEWMA of the time series
    fbewma_series = fbewma(input_data, span)

    # Create a condition to identify outliers based on the absolute difference
    # between input_data and fbewma_series
    delta_cond = np.abs(input_data - fbewma_series) > delta

    # Set values to NaN where the condition is True
    return input_data.where(~delta_cond, np.nan)


def remove_spikes(
    input_data: pd.Series, span: int, low_clip: float, high_clip: float, delta: float
) -> pd.Series:
    """Remove spikes.

    Remove spikes from a time series data using a combination of clipping and
    interpolation.

    Parameters
    ----------
    input_data : pandas.Series
        Input time series data.
    span : int
        Span parameter for exponential weighting used in outlier detection.
    low_clip : float
        Lower bound for clipping. Values less than this will be set to NaN.
    high_clip : float
        Upper bound for clipping. Values greater than this will be set to NaN.
    delta : float
        Threshold for identifying outliers. Values greater than this threshold
        will be considered spikes.

    Returns
    -------
    pandas.Series
        A Series containing the time series with spikes removed with the same
        index as the input Series.
    """
    # Clip values in the input data within the specified range
    clipped = clip(input_data, low_clip, high_clip)

    # Remove outliers using the remove_outliers function
    gaps_series = remove_outliers(clipped, span, delta)

    # Could use pandas' .interpolate() on the Series
    # interp_series = gaps_series.interpolate()

    return gaps_series


def remove_range(
    input_series: pd.Series,
    from_date: str | None,
    to_date: str | None,
    min_gap_length: int = 1,
    insert_gaps: str = "none",
):
    """
    Remove data from series in given range.

    Returns the input series without data between from_date and to_date
    inclusive.

    A None to_date will remove all data since the from_date (and vice versa).
    A double None for to_date/from_date removes all data.

    Inserts gaps or not depending on insert_gaps

    Parameters
    ----------
    input_series : pd.Series
        The series to have a section removed
    from_date : str | None
        Start of removed section
    to_date : str | None
        End of removed section
    min_gap_length : int
        Will insert gaps based on insert_gaps strategy if missing more data points than
        min_gap_length in a row.
    insert_gaps : str
        If "all" will insert np.NaN at every missing point.
        If "start" will insert np.NaN only at from_date.
        If "end" will insert np.NaN only at to_date.
        If "none" will insert no np.NaN values, and remove all timestamps completely.

    Returns
    -------
    pd.Series
        The series with relevant slice removed
    """
    input_series = input_series.copy()
    slice_to_remove = input_series.loc[from_date:to_date]

    if len(slice_to_remove) >= min_gap_length:
        if insert_gaps == "all":
            series_to_return = input_series.copy()
            series_to_return.loc[from_date:to_date] = np.NaN
        else:
            series_to_return = input_series.drop(slice_to_remove.index)
            if insert_gaps == "start":
                start_idx = slice_to_remove.index[0]
                series_to_return[start_idx] = np.NaN
            elif insert_gaps == "end":
                end_idx = slice_to_remove.index[-1]
                series_to_return[end_idx] = np.NaN
            elif insert_gaps == "none":
                pass
            else:
                raise ValueError(
                    f"Unknown value for argument {insert_gaps}. Choose one of 'all', 'start', 'end', 'none'."
                )
    else:
        series_to_return = input_series.drop(slice_to_remove.index)
    return series_to_return.sort_index()


def trim_series(std_series: pd.Series, check_series: pd.Series) -> pd.Series:
    """
    Remove end of std series to match check series.

    All data after the last entry in check_series is presumed to be unchecked,
    so that data is removed from the std_series

    If check_series is empty, returns the entire std_series

    Parameters
    ----------
    std_series : pd.Series
        The series to be trimmed
    check_series : pd.Series | pd.DataFrame
        Indicates the end of the usable data

    Returns
    -------
    pd.Series
        std_series with the unchecked elements trimmed
    """
    if check_series.empty:
        return std_series
    else:
        last_check_date = check_series.index[-1]
        return std_series.loc[:last_check_date]


def flatline_value_remover(
    series: pd.Series,
    span: int = 3,
):
    """
    Remove repeated (flatlined) values in a series.

    Examines the data to see if any values are exactly repeated over a period.
    Where values exactly repeat it probably indicates a broken instrument.
    Replaces all values after the first with NaN.
    Uses math.isclose() to measure float "equality"

    Parameters
    ----------
    series : pd.Series
        Data to examine for flatlined values
    span : int
        Amount of allowed repeated values in a row before duplicates are removed

    Returns
    -------
    pd.Series
        Data with the flatlined values NaN'ed
    """
    # pandas bad day
    consecutive_values = (
        series.groupby((series != series.shift()).cumsum()).cumcount() + 1
    )
    working_step = consecutive_values.loc[~consecutive_values.between(2, span)]
    length_filter = working_step.reindex(consecutive_values.index).bfill()
    filtered_data = pd.Series(series[length_filter < span])
    return filtered_data.reindex(series.index)
