"""Processor class."""

import warnings
from functools import wraps

import numpy as np
import pandas as pd
from annalist.annalist import Annalist
from annalist.decorators import ClassLogger
from hilltoppy import Hilltop

from hydrobot import (
    data_acquisition,
    data_sources,
    evaluator,
    filters,
    plotter,
    xml_data_structure,
)

annalizer = Annalist()

DEFAULTS = {
    "high_clip": 20000,
    "low_clip": 0,
    "delta": 1000,
    "span": 10,
    "gap_limit": 12,
    "max_qc": np.NaN,
}

MOWSECS_OFFSET = 946771200


def stale_warning(method):
    """Decorate dangerous functions.

    Check whether the data is stale, and warn user if so.
    Warning will then take input form user to determine whether to proceed or cancel.
    Cancelling will return a null function, which returns None with no side effects no
    matter what the input

    Currently broken

    Parameters
    ----------
    method : function
        A function that might have some problems if the parameters have been changed
        but the data hasn't been updated

    Returns
    -------
    function
        null function if warning is heeded, otherwise
    """

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        if self._stale:
            warnings.warn(
                "Warning: a key parameter of the data has changed but the data itself "
                "has not been reloaded.",
                stacklevel=2,
            )
            while True:
                user_input = input("Do you want to continue? y/n: ")

                if user_input.lower() in ["y", "ye", "yes"]:
                    print("Continuing")
                    return method(self, *method_args, **method_kwargs)
                if user_input.lower() in ["n", "no"]:
                    print("Function cancelled")
                    return lambda *x: None
                print("Type y or n (or yes or no, or even ye, all ye who enter here)")
        else:
            return method(self, *method_args, **method_kwargs)

    return _impl


class Processor:
    """
    A class used to process data from a Hilltop server.

    Attributes
    ----------
    _defaults : dict
        The default settings.
    _site : str
        The site to be processed.
    _standard_measurement : str
        The standard measurement to be processed.
    _check_measurement : str
        The measurement to be checked.
    _base_url : str
        The base URL of the Hilltop server.
    _standard_hts : str
        The standard Hilltop service.
    _check_hts : str
        The Hilltop service to be checked.
    _frequency : str
        The frequency of the data.
    _from_date : str
        The start date of the data.
    _to_date : str
        The end date of the data.
    _measurement : Measurement
        The measurement data.
    _stale : bool
        The stale status of the data.
    _no_data : bool
        The no data status of the data.
    _standard_series : pd.Series
        The standard series data.
    _check_series : pd.Series
        The check series data.
    _quality_series : pd.Series
        The quality series data.

    Methods
    -------
    import_data():
        Loads the data for the first time.
    """

    @ClassLogger  # type: ignore
    def __init__(
        self,
        base_url: str,
        site: str,
        standard_hts: str,
        standard_measurement: str,
        frequency: str,
        from_date: str | None = None,
        to_date: str | None = None,
        check_hts: str | None = None,
        check_measurement: str | None = None,
        defaults: dict | None = None,
        **kwargs,
    ):
        """
        Constructs all the necessary attributes for the Processor object.

        Parameters
        ----------
        base_url : str
            The base URL of the Hilltop server.
        site : str
            The site to be processed.
        standard_hts : str
            The standard Hilltop service.
        standard_measurement : str
            The standard measurement to be processed.
        frequency : str
            The frequency of the data.
        from_date : str, optional
            The start date of the data (default is None).
        to_date : str, optional
            The end date of the data (default is None).
        check_hts : str, optional
            The Hilltop service to be checked (default is None).
        check_measurement : str, optional
            The measurement to be checked (default is None).
        defaults : dict, optional
            The default settings (default is None).
        kwargs : dict
            Additional keyword arguments.
        """
        if defaults is None:
            self._defaults = DEFAULTS
        else:
            self._defaults = defaults
        if check_hts is None:
            check_hts = standard_hts
        if check_measurement is None:
            check_measurement = standard_measurement

        standard_hilltop = Hilltop(base_url, standard_hts, **kwargs)
        check_hilltop = Hilltop(base_url, check_hts, **kwargs)
        if (
            site in standard_hilltop.available_sites
            and site in check_hilltop.available_sites
        ):
            self._site = site
        else:
            raise ValueError(
                f"Site '{site}' not found for both base_url and hts combos."
                f"Available sites in standard_hts are: "
                f"{[s for s in standard_hilltop.available_sites]}"
                f"Available sites in check_hts are: "
                f"{[s for s in check_hilltop.available_sites]}"
            )

        standard_measurement_list = standard_hilltop.get_measurement_list(site)
        if standard_measurement in list(standard_measurement_list.MeasurementName):
            self._standard_measurement = standard_measurement
        else:
            raise ValueError(
                f"Standard measurement '{standard_measurement}' not found at "
                f"site '{site}'. "
                "Available measurements are "
                f"{list(standard_measurement_list.MeasurementName)}"
            )
        check_measurement_list = check_hilltop.get_measurement_list(site)
        if check_measurement in list(check_measurement_list.MeasurementName):
            self._check_measurement = check_measurement
        else:
            raise ValueError(
                f"Check measurement '{check_measurement}' not found at site '{site}'. "
                "Available measurements are "
                f"{list(check_measurement_list.MeasurementName)}"
            )

        self._base_url = base_url
        self._standard_hts = standard_hts
        self._check_hts = check_hts
        self._frequency = frequency
        self._from_date = from_date
        self._to_date = to_date
        self._qc_evaluator = data_sources.get_qc_evaluator(standard_measurement)

        self._stale = True
        self._no_data = True
        self._standard_series = pd.Series({})
        self._check_series = pd.Series({})
        self._quality_series = pd.DataFrame({})

        # Load data for the first time
        self.import_data()

    @property
    def standard_measurement(self):  # type: ignore
        """str: The site to be processed."""
        return self._standard_measurement

    @property
    def site(self):  # type: ignore
        """
        str: The site to be processed.

        Setting this property will mark the data as stale.
        """
        return self._site

    # @ClassLogger  # type: ignore
    # @site.setter
    # def site(self, value):
    #     self._site = value
    #     self._stale = True

    @property
    def from_date(self):  # type: ignore
        """
        str: The start date of the data.

        Setting this property will mark the data as stale.
        """
        return self._from_date

    # @ClassLogger  # type: ignore
    # @from_date.setter
    # def from_date(self, value):
    #     self._from_date = value
    #     self._stale = True

    @property
    def to_date(self):  # type: ignore
        """
        str: The end date of the data.

        Setting this property will mark the data as stale.
        """
        return self._to_date

    # @ClassLogger  # type: ignore
    # @to_date.setter
    # def to_date(self, value):
    #     self._to_date = value
    #     self._stale = True

    @property
    def frequency(self):  # type: ignore
        """
        str: The frequency of the data.

        Setting this property will mark the data as stale.
        """
        return self._frequency

    # @ClassLogger  # type: ignore
    # @frequency.setter
    # def frequency(self, value):
    #     self._frequency = value
    #     self._stale = True

    @property
    def base_url(self):  # type: ignore
        """
        str: The base URL of the Hilltop server.

        Setting this property will mark the data as stale.
        """
        return self._base_url

    # @ClassLogger  # type: ignore
    # @base_url.setter
    # def base_url(self, value):
    #     self._base_url = value
    #     self._stale = True

    @property
    def standard_hts(self):  # type: ignore
        """
        str: The standard Hilltop service.

        Setting this property will mark the data as stale.
        """
        return self._standard_hts

    # @ClassLogger  # type: ignore
    # @standard_hts.setter
    # def standard_hts(self, value):
    #     self._standard_hts = value
    #     self._stale = True

    @property
    def check_hts(self):  # type: ignore
        """
        str: The Hilltop service to be checked.

        Setting this property will mark the data as stale.
        """
        return self._check_hts

    # @ClassLogger  # type: ignore
    # @check_hts.setter
    # def check_hts(self, value):
    #     self._check_hts = value
    #     self._stale = True

    @property
    def qc_evaluator(self):  # type: ignore
        """
        Measurement: The measurement data.

        Setting this property will mark the data as stale.
        """
        return self._qc_evaluator

    # @ClassLogger  # type: ignore
    # @measurement.setter
    # def measurement(self, value):
    #     self._qc_evaluator = value
    #     self._stale = True

    @property
    def defaults(self):  # type: ignore
        """
        dict: The default settings.

        Setting this property will mark the data as stale.
        """
        return self._defaults

    # @ClassLogger  # type: ignore
    # @defaults.setter
    # def defaults(self, value):
    #     self._defaults = value
    #     self._stale = True

    @property  # type: ignore
    def standard_series(self):  # type: ignore
        """pd.Series: The standard series data."""
        return self._standard_series

    @ClassLogger  # type: ignore
    @standard_series.setter
    def standard_series(self, value):
        self._standard_series = value

    @property
    def check_series(self):  # type: ignore
        """pd.Series: The series containing check data."""
        return self._check_series

    @ClassLogger  # type: ignore
    @check_series.setter
    def check_series(self, value):
        self._check_series = value

    @property
    def quality_series(self):  # type: ignore
        """
        pd.Series: The quality series data.

        Setting this property will mark the data as stale.
        """
        return self._quality_series

    @ClassLogger  # type: ignore
    @quality_series.setter
    def quality_series(self, value):
        self._quality_series = value
        self._stale = True

    @ClassLogger
    def import_range(
        self,
        from_date: str | None,
        to_date: str | None,
        standard: bool = True,
        check: bool = True,
        quality: bool = False,
    ):
        """
        Load Raw Data from Hilltop within a specified date range.

        Parameters
        ----------
        from_date : str or None
            The start date for data import. If None, the earliest available data will
            be used.
        to_date : str or None
            The end date for data import. If None, the latest available data will be
            used.
        standard : bool, optional
            Whether to import standard data, by default True.
        check : bool, optional
            Whether to import check data, by default True.
        quality : bool, optional
            Whether to import quality data, by default False.

        Returns
        -------
        None

        Notes
        -----
        This method retrieves raw data from Hilltop for the specified date range and
        updates the internal series data in the Processor instance. The data can be
        imported for standard, check, and quality measurements separately.

        Raises
        ------
        ValueError
            If the specified data type is not found or if no data is found for the
            given range.

        Warnings
        --------
        This method modifies the internal series data based on the retrieved Hilltop
        data.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.import_range(from_date="2022-01-01", to_date="2022-12-31", \
            standard=True)
        """
        # If this is the first data import, initialize with an empty dict
        if self._no_data:
            self.raw_data_dict = {}

        # These will become the keys for the raw_data_dict
        ts_types = ["standard", "check", "qual"]

        # Boolean flag arguments for the three types of timeseries
        inc_types = [standard, check, quality]

        # Dropping keys from ts_types if corresponding flag is set to False
        import_types = [ts for ts, inc in zip(ts_types, inc_types) if inc]

        # Iterating through all the data_types to be imported here
        for data_type in import_types:
            # Setting up some defaults for each timeseries type
            if data_type == "standard":
                req_type = "Standard"
                ds_type = "StdSeries"
                if isinstance(self._standard_series, pd.Series):
                    curr_series = self._standard_series
                else:
                    warnings.warn(
                        "Existing Standard Series should be pandas.Series, but found "
                        f"{type(self._standard_series)}. Setting to empty pd.Series",
                        stacklevel=1,
                    )
                    curr_series = pd.Series({})
                data_series = pd.Series({})
            elif data_type == "check":
                req_type = "Check"
                ds_type = "CheckSeries"
                # Check data has many fields, so it comes in as a DataFrame
                data_series = pd.DataFrame({})
                if isinstance(self._check_series, pd.DataFrame):
                    curr_series = self._check_series
                else:
                    warnings.warn(
                        "Existing Check Series should be pandas.DataFrame, but found"
                        f" {type(self._check_series)}. Setting to empty pd.DataFrame",
                        stacklevel=1,
                    )
                    curr_series = pd.DataFrame({})
            elif data_type == "qual":
                req_type = "Quality"
                ds_type = "StdQualSeries"
                data_series = pd.Series({})
                if isinstance(self._quality_series, pd.Series):
                    curr_series = self._quality_series
                else:
                    warnings.warn(
                        "Existing Standard Series should be pandas.Series, but found "
                        f"{type(self._standard_series)}. Setting to empty pd.Series",
                        stacklevel=1,
                    )
                    curr_series = pd.Series({})
            else:
                raise ValueError(
                    "I believe this error is unreachable, but Pyright disagrees"
                )

            blob_list = data_acquisition.get_data(
                self._base_url,
                self._standard_hts,
                self._site,
                self._standard_measurement,
                from_date,
                to_date,
                tstype=req_type,
            )

            blob_found = False
            # Iterating through all the blobs to find the timeseries for this data_type
            for blob in blob_list:
                if (blob.data_source.name == self._standard_measurement) and (
                    blob.data_source.ts_type == ds_type
                ):
                    # Found it. Now we extract it.
                    blob_found = True

                    # This could be a pd.Series or a pd.DataFrame
                    data_series = blob.data.timeseries
                    if self._no_data:
                        self.raw_data_dict[data_type] = blob
                    if not data_series.empty:
                        raw_time = data_series.index.astype("int64")
                        mowsec_time = raw_time.map(lambda x: x - MOWSECS_OFFSET)
                        timestamps = mowsec_time.map(
                            lambda x: pd.Timestamp(x, unit="s")
                            if x is not None
                            else None
                        )
                        data_series.index = pd.to_datetime(timestamps)
            if not blob_found:
                raise ValueError(f"{req_type} Data Not Found")

            if data_type == "standard":
                insert_series = data_series.asfreq(self._frequency, method="bfill")
            else:
                insert_series = data_series

            if not curr_series.empty:
                slice_to_remove = curr_series.loc[
                    insert_series.index[0] : insert_series.index[-1]
                ]
                curr_series = curr_series.drop(slice_to_remove.index)

            # Pandas doesn't like concatting possibly empty series anymore.
            # Test before upgrading pandas for release.
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                # Check for performance at some point
                if data_type == "standard":
                    self.standard_series = pd.concat(
                        [
                            curr_series,
                            insert_series,
                        ]
                    ).sort_index()
                if data_type == "check":
                    self.check_series = pd.concat(
                        [
                            curr_series,
                            insert_series,
                        ]
                    ).sort_index()
                if data_type == "qual":
                    self.quality_series = pd.concat(
                        [
                            curr_series,
                            insert_series,
                        ]
                    ).sort_index()

    def import_data(
        self,
        standard: bool = True,
        check: bool = True,
        quality: bool = False,
    ):
        """
        Import data using the class parameter range.

        Parameters
        ----------
        standard : bool, optional
            Whether to import standard data, by default True.
        check : bool, optional
            Whether to import check data, by default True.
        quality : bool, optional
            Whether to import quality data, by default False.

        Returns
        -------
        None

        Notes
        -----
        This method imports data for the specified date range, using the class
        parameters `_from_date` and `_to_date`. It updates the internal series data in
        the Processor instance for standard, check, and quality measurements
        separately.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.set_date_range("2022-01-01", "2022-12-31")
        >>> processor.import_data(standard=True, check=True)
        >>> processor.stale
        False
        """
        self._standard_series = pd.Series({})
        self._check_series = pd.DataFrame({})
        self._quality_series = pd.Series({})
        self.import_range(self._from_date, self._to_date, standard, check, quality)
        self._stale = False

    # @stale_warning  # type: ignore
    @ClassLogger
    def gap_closer(self, gap_limit: int | None = None):
        """
        Close small gaps in the standard series.

        Parameters
        ----------
        gap_limit : int or None, optional
            The maximum number of consecutive missing values to close, by default None.
            If None, the gap limit from the class defaults is used.

        Returns
        -------
        None

        Notes
        -----
        This method closes small gaps in the standard series by replacing consecutive
        missing values with interpolated or backfilled values. The gap closure is
        performed using the evaluator.small_gap_closer function.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.gap_closer(gap_limit=5)
        >>> processor.standard_series
        <updated standard series with closed gaps>
        """
        if gap_limit is None:
            gap_limit = int(self._defaults["gap_limit"])
        self.standard_series = evaluator.small_gap_closer(
            self._standard_series, gap_limit=gap_limit
        )

    # @stale_warning  # type: ignore
    @ClassLogger
    def quality_encoder(
        self, gap_limit: int | None = None, max_qc: int | float | None = None
    ):
        """
        Encode quality information in the quality series.

        Parameters
        ----------
        gap_limit : int or None, optional
            The maximum number of consecutive missing values to consider as gaps, by
            default None.
            If None, the gap limit from the class defaults is used.

        Returns
        -------
        None

        Notes
        -----
        This method encodes quality information in the quality series based on the
        provided standard series, check series, and measurement information. It uses
        the evaluator.quality_encoder function to determine the quality flags for the
        data.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.quality_encoder(gap_limit=5)
        >>> processor.quality_series
        <updated quality series with encoded quality flags>
        """
        if gap_limit is None:
            gap_limit = int(self._defaults["gap_limit"])
        if max_qc is None:
            max_qc = self._defaults["max_qc"] if "max_qc" in self._defaults else np.NaN
        self.quality_series = evaluator.quality_encoder(
            self._standard_series,
            pd.Series(self._check_series["Check Guage Total"]),
            self._qc_evaluator,
            gap_limit=gap_limit,
            max_qc=max_qc,
        )

    # @stale_warning  # type: ignore
    @ClassLogger
    def clip(self, low_clip: float | None = None, high_clip: float | None = None):
        """
        Clip data within specified low and high values.

        Parameters
        ----------
        low_clip : float or None, optional
            The lower bound for clipping, by default None.
            If None, the low clip value from the class defaults is used.
        high_clip : float or None, optional
            The upper bound for clipping, by default None.
            If None, the high clip value from the class defaults is used.

        Returns
        -------
        None

        Notes
        -----
        This method clips the data in both the standard and check series within the
        specified low and high values. It uses the filters.clip function for the actual
        clipping process.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.clip(low_clip=0, high_clip=100)
        >>> processor.standard_series
        <clipped standard series within the specified range>
        >>> processor.check_series
        <clipped check series within the specified range>
        """
        if low_clip is None:
            low_clip = float(self._defaults["low_clip"])
        if high_clip is None:
            high_clip = float(self._defaults["high_clip"])

        self.standard_series = filters.clip(self._standard_series, low_clip, high_clip)
        self.check_series = filters.clip(
            pd.Series(self._check_series["Check Guage Total"]), low_clip, high_clip
        )

    # @stale_warning  # type: ignore
    @ClassLogger
    def remove_outliers(self, span: int | None = None, delta: float | None = None):
        """
        Remove outliers from the data.

        Parameters
        ----------
        span : int or None, optional
            The span parameter for smoothing, by default None.
            If None, the span value from the class defaults is used.
        delta : float or None, optional
            The delta parameter for identifying outliers, by default None.
            If None, the delta value from the class defaults is used.

        Returns
        -------
        None

        Notes
        -----
        This method removes outliers from the standard series using the specified
        span and delta values. It utilizes the filters.remove_outliers function for
        the actual outlier removal process.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.remove_outliers(span=10, delta=2.0)
        >>> processor.standard_series
        <standard series with outliers removed>
        """
        if span is None:
            span = int(self._defaults["span"])
        if delta is None:
            delta = float(self._defaults["delta"])

        if isinstance(self._standard_series, pd.Series):
            self.standard_series = filters.remove_outliers(
                self._standard_series, span, delta
            )
        else:
            raise TypeError(
                "Standard Series should be pd.Series, "
                f"found {type(self._standard_series)}."
            )

    # @stale_warning  # type: ignore
    @ClassLogger
    def remove_spikes(
        self,
        low_clip: float | None = None,
        high_clip: float | None = None,
        span: int | None = None,
        delta: float | None = None,
    ):
        """
        Remove spikes from the data.

        Parameters
        ----------
        low_clip : float or None, optional
            The lower clipping threshold, by default None.
            If None, the low_clip value from the class defaults is used.
        high_clip : float or None, optional
            The upper clipping threshold, by default None.
            If None, the high_clip value from the class defaults is used.
        span : int or None, optional
            The span parameter for smoothing, by default None.
            If None, the span value from the class defaults is used.
        delta : float or None, optional
            The delta parameter for identifying spikes, by default None.
            If None, the delta value from the class defaults is used.

        Returns
        -------
        None

        Notes
        -----
        This method removes spikes from the standard series using the specified
        parameters. It utilizes the filters.remove_spikes function for the actual
        spike removal process.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.remove_spikes(low_clip=10, high_clip=20, span=5, delta=2.0)
        >>> processor.standard_series
        <standard series with spikes removed>
        """
        if low_clip is None:
            low_clip = float(self._defaults["low_clip"])
        if high_clip is None:
            high_clip = float(self._defaults["high_clip"])
        if span is None:
            span = int(self._defaults["span"])
        if delta is None:
            delta = float(self._defaults["delta"])

        if isinstance(self._standard_series, pd.Series):
            self.standard_series = filters.remove_spikes(
                self._standard_series, span, low_clip, high_clip, delta
            )
        else:
            raise TypeError(
                "Standard Series should be pd.Series,"
                f"found {type(self._standard_series)}"
            )

    @ClassLogger
    def delete_range(
        self,
        from_date,
        to_date,
        tstype_standard=True,
        tstype_check=False,
        tstype_quality=False,
    ):
        """
        Delete a range of data from specified time series types.

        Parameters
        ----------
        from_date : str
            The start date of the range to delete.
        to_date : str
            The end date of the range to delete.
        tstype_standard : bool, optional
            Flag to delete data from the standard series, by default True.
        tstype_check : bool, optional
            Flag to delete data from the check series, by default False.
        tstype_quality : bool, optional
            Flag to delete data from the quality series, by default False.

        Returns
        -------
        None

        Notes
        -----
        This method deletes a specified range of data from the selected time series
        types. The range is defined by the `from_date` and `to_date` parameters.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.delete_range(from_date="2022-01-01", to_date="2022-12-31", \
                tstype_standard=True)
        >>> processor.standard_series
        <standard series with specified range deleted>
        >>> processor.delete_range(from_date="2022-01-01", to_date="2022-12-31", \
                tstype_check=True)
        >>> processor.check_series
        <check series with specified range deleted>
        """
        if tstype_standard:
            self.standard_series = filters.remove_range(
                self._standard_series, from_date, to_date
            )
        if tstype_check:
            self.check_series = filters.remove_range(
                self._check_series, from_date, to_date
            )
        if tstype_quality:
            self.quality_series = filters.remove_range(
                self._quality_series, from_date, to_date
            )

    @ClassLogger
    def insert_missing_nans(self):
        """
        Set the data to the correct frequency, filled with NaNs as appropriate.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method adjusts the time series data to the correct frequency,
        filling missing values with NaNs as appropriate. It modifies the
        standard series in-place.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.insert_missing_nans()
        >>> processor.standard_series
        <standard series with missing values filled with NaNs>
        """
        self.standard_series = self._standard_series.asfreq(self._frequency)

    @ClassLogger
    def data_exporter(self, file_location, trimmed=True):
        """
        Export data to CSV file.

        [DEPRECATED]

        Parameters
        ----------
        file_location : str
            The file path where the CSV file will be saved.
        trimmed : bool, optional
            If True, export trimmed data; otherwise, export the full data.
            Default is True.

        Returns
        -------
        None

        Notes
        -----
        This method exports data to a CSV file. It is deprecated and may be
        removed in future releases. Use with caution.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.data_exporter("output.csv", trimmed=True)
        >>> # Check the generated CSV file at 'output.csv'
        """
        if (
            isinstance(self._standard_series, pd.Series)
            and isinstance(self._check_series, pd.DataFrame)
            and isinstance(self._quality_series, pd.Series)
        ):
            if trimmed:
                std_series = filters.trim_series(
                    self._standard_series,
                    self._check_series,
                )
            else:
                std_series = self._standard_series
            data_sources.series_export_to_csv(
                file_location,
                self._site,
                self._qc_evaluator.name,
                std_series,
                self._check_series,
                self._quality_series,
            )
        else:
            raise TypeError(
                "Standard Series should be pd.Series, "
                f"found {type(self._standard_series)}"
            )
        data_sources.hilltop_export(
            file_location,
            self._site,
            self._qc_evaluator.name,
            std_series,
            self._check_series,
            self._quality_series,
        )

    def diagnosis(self):
        """
        Provide a diagnosis of the data.

        Returns
        -------
        None

        Notes
        -----
        This method analyzes the state of the data, including the standard,
        check, and quality series. It provides diagnostic information about
        the data distribution, gaps, and other relevant characteristics.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.import_data()
        >>> processor.diagnosis()
        >>> # View diagnostic information about the data.
        """
        evaluator.diagnose_data(
            self._standard_series,
            self._check_series,
            self._quality_series,
            self._frequency,
        )

    def plot_qc_series(self, show=True):
        """Implement qc_plotter()."""
        plotter.qc_plotter(
            self._standard_series,
            self._check_series,
            self._quality_series,
            self._frequency,
            show=show,
        )

    def plot_gaps(self, span=None, show=True):
        """
        Plot gaps in the data.

        Parameters
        ----------
        span : int | None, optional
            Size of the moving window for identifying gaps. If None, the default
            behavior is used. Default is None.
        show : bool, optional
            Whether to display the plot. If True, the plot is displayed; if False,
            the plot is generated but not displayed. Default is True.

        Returns
        -------
        None

        Notes
        -----
        This method utilizes the gap_plotter function to visualize gaps in the
        standard series data. Gaps are identified based on the specified span.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.import_data()
        >>> processor.plot_gaps(span=10, show=True)
        >>> # Display a plot showing gaps in the standard series.
        """
        if span is None:
            plotter.gap_plotter(self._standard_series, show=show)
        else:
            plotter.gap_plotter(self._standard_series, span, show=show)

    def plot_checks(self, span=None, show=True):
        """
        Plot checks against the standard series data.

        Parameters
        ----------
        span : int | None, optional
            Size of the moving window for smoothing the plot. If None, the default
            behavior is used. Default is None.
        show : bool, optional
            Whether to display the plot. If True, the plot is displayed; if False,
            the plot is generated but not displayed. Default is True.

        Returns
        -------
        None

        Notes
        -----
        This method utilizes the check_plotter function to visualize checks against
        the standard series data. The plot includes both the standard and check series.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.import_data()
        >>> processor.plot_checks(span=10, show=True)
        >>> # Display a plot comparing checks to the standard series.
        """
        if span is None:
            plotter.check_plotter(self._standard_series, self._check_series, show=show)
        else:
            plotter.check_plotter(
                self._standard_series, self._check_series, span, show=show
            )

    def to_xml_data_structure(self):
        """
        Convert Processor object data to a list of XML data structures.

        Returns
        -------
        list of xml_data_structure.DataSourceBlob
            List of DataSourceBlob instances representing the data in the Processor
            object.

        Notes
        -----
        This method converts the data in the Processor object, including standard,
        check, and quality series, into a list of DataSourceBlob instances. Each
        DataSourceBlob contains information about the site, data source, and associated
        data.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.import_data()
        >>> xml_data_list = processor.to_xml_data_structure()
        >>> # Convert Processor data to a list of XML data structures.
        """
        data_blob_list = []

        for dtype, raw_blob in self.raw_data_dict.items():
            if raw_blob.item_info is not None:
                item_info_list = []
                for i, info in enumerate(raw_blob.item_info):
                    item_info = xml_data_structure.ItemInfo(
                        item_number=i,
                        item_name=info.item_name,
                        item_format=info.item_format,
                        divisor=info.divisor,
                        units=info.units,
                        format=info.format,
                    )
                    item_info_list += [item_info]
            else:
                item_info_list = []

            data_source = xml_data_structure.DataSource(
                name=self._standard_measurement,
                num_items=len(item_info_list),
                ts_type=raw_blob.data_source.ts_type,
                data_type=raw_blob.data_source.data_type,
                interpolation=raw_blob.data_source.data_type,
                item_format=raw_blob.data_source.item_format,
                item_info=raw_blob.data_source.item_format,
            )

            if dtype == "standard":
                if isinstance(self._standard_series, pd.Series):
                    timeseries = self._standard_series
                else:
                    raise TypeError(
                        "Standard Series should be pd.Series, "
                        f"found {type(self._standard_series)}"
                    )
            elif dtype == "check":
                if isinstance(self._check_series, pd.DataFrame):
                    timeseries = self._check_series
                else:
                    raise TypeError(
                        "Standard Series should be pd.DataFrame, "
                        f"found {type(self._check_series)}"
                    )
            elif dtype == "qual":
                if isinstance(self._quality_series, pd.Series):
                    timeseries = self._quality_series
                else:
                    raise TypeError(
                        "Standard Series should be pd.Series, "
                        f"found {type(self._quality_series)}"
                    )
            else:
                raise ValueError("I don't think anyone will ever get this error.")
            data = xml_data_structure.Data(
                date_format=raw_blob.data.date_format,
                num_items=raw_blob.data.num_items,
                timeseries=timeseries,
            )

            data_blob = xml_data_structure.DataSourceBlob(
                site_name=raw_blob.site_name,
                data_source=data_source,
                data=data,
            )

            data_blob_list += [data_blob]
        return data_blob_list
