"""Main module."""

import pandas as pd
from annalist.annalist import Annalist
from hilltoppy.utils import build_url, get_hilltop_xml

from hydrobot.xml_data_structure import parse_xml

annalizer = Annalist()


def get_data(
    base_url,
    hts,
    site,
    measurement,
    from_date,
    to_date,
    tstype="Standard",
):
    """Acquire time series data from a web service and return it as a DataFrame.

    Parameters
    ----------
    base_url : str
        The base URL of the web service.
    hts : str
        The Hilltop Time Series (HTS) identifier.
    site : str
        The site name or location.
    measurement : str
        The type of measurement to retrieve.
    from_date : str
        The start date and time for data retrieval
        in the format 'YYYY-MM-DD HH:mm'.
    to_date : str
        The end date and time for data retrieval
        in the format 'YYYY-MM-DD HH:mm'.
    tstype : str
        Type of data that is sought
        (default is Standard, can be Standard, Check, or Quality)

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the acquired time series data.
    """
    url = build_url(
        base_url,
        hts,
        "GetData",
        site=site,
        measurement=measurement,
        from_date=from_date,
        to_date=to_date,
        tstype=tstype,
    )

    hilltop_xml = get_hilltop_xml(url)

    data_object = parse_xml(hilltop_xml)

    return data_object


def get_series(
    base_url,
    hts,
    site,
    measurement,
    from_date,
    to_date,
    tstype="Standard",
):
    """Pack data from det_data as a pd.Series.

    Parameters
    ----------
    base_url : str
        The base URL of the web service.
    hts : str
        The Hilltop Time Series (HTS) identifier.
    site : str
        The site name or location.
    measurement : str
        The type of measurement to retrieve.
    from_date : str
        The start date and time for data retrieval
        in the format 'YYYY-MM-DD HH:mm'.
    to_date : str
        The end date and time for data retrieval
        in the format 'YYYY-MM-DD HH:mm'.
    tstype : str
        Type of data that is sought
        (default 'Standard, can be Standard, Check, or Quality)

    Returns
    -------
    pandas.Series
        A pd.Series containing the acquired time series data.
    """
    data_object = get_data(
        base_url,
        hts,
        site,
        measurement,
        from_date,
        to_date,
        tstype,
    )
    data = data_object[0].data.timeseries
    if not data.empty:
        data.index.name = "Time"
        data.name = "Value"
        mowsecs_offset = 946771200
        timestamps = data.index.map(
            lambda x: pd.Timestamp(int(x) - mowsecs_offset, unit="s")
        )
        data.index = pd.to_datetime(timestamps)
    else:
        data = pd.Series({})
    return data
