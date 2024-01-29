"""Processor class."""

import re
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
    utils,
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
    """A class used to process data from a Hilltop server."""

    @ClassLogger  # type: ignore
    def __init__(
        self,
        base_url: str,
        site: str,
        standard_hts: str,
        standard_measurement_name: str,
        frequency: str,
        from_date: str | None = None,
        to_date: str | None = None,
        check_hts: str | None = None,
        check_measurement_name: str | None = None,
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
        if check_measurement_name is None:
            check_measurement_name = standard_measurement_name

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

        available_standard_measurements = standard_hilltop.get_measurement_list(site)
        if standard_measurement_name in list(
            available_standard_measurements.MeasurementName
        ):
            self._standard_measurement_name = standard_measurement_name
        else:
            raise ValueError(
                f"Standard measurement name '{standard_measurement_name}' not found at "
                f"site '{site}'. "
                "Available measurements are "
                f"{list(available_standard_measurements.MeasurementName)}"
            )
        available_check_measurements = check_hilltop.get_measurement_list(site)
        self._check_measurement_name = check_measurement_name
        matches = re.search(r"([^\[\n]+)(\[(.+)\])?", check_measurement_name)

        if matches is not None:
            self.check_item_name = matches.groups()[0].strip(" ")
            self.check_data_source_name = matches.groups()[2]
            if self.check_data_source_name is None:
                self.check_data_source_name = self.check_item_name
        if self._check_measurement_name not in list(
            available_check_measurements.MeasurementName
        ):
            raise ValueError(
                f"Check measurement name '{self._check_measurement_name}' "
                f"not found at site '{site}'. "
                "Available measurements are "
                f"{list(available_check_measurements.MeasurementName)}"
            )

        self._base_url = base_url
        self._standard_hts = standard_hts
        self._check_hts = check_hts
        self._frequency = frequency
        self._from_date = from_date
        self._to_date = to_date
        self._quality_code_evaluator = data_sources.get_qc_evaluator(
            standard_measurement_name
        )
        self._stale = True
        self._no_data = True
        self._standard_series = pd.Series({})
        self._check_series = pd.Series({})
        self._check_data = pd.DataFrame({})
        self._quality_series = pd.Series({})

        self.raw_data_dict = {
            "standard": {},
            "quality": {},
            "check": {},
        }

        # Load data for the first time
        self.import_data(from_date=self.from_date, to_date=self.to_date)

    @property
    def standard_measurement_name(self):  # type: ignore
        """str: The site to be processed."""
        return self._standard_measurement_name

    @property
    def site(self):  # type: ignore
        """
        str: The site to be processed.

        Setting this property will mark the data as stale.
        """
        return self._site

    @property
    def from_date(self):  # type: ignore
        """
        str: The start date of the data.

        Setting this property will mark the data as stale.
        """
        return self._from_date

    @property
    def to_date(self):  # type: ignore
        """
        str: The end date of the data.

        Setting this property will mark the data as stale.
        """
        return self._to_date

    @property
    def frequency(self):  # type: ignore
        """
        str: The frequency of the data.

        Setting this property will mark the data as stale.
        """
        return self._frequency

    @property
    def base_url(self):  # type: ignore
        """
        str: The base URL of the Hilltop server.

        Setting this property will mark the data as stale.
        """
        return self._base_url

    @property
    def standard_hts(self):  # type: ignore
        """
        str: The standard Hilltop service.

        Setting this property will mark the data as stale.
        """
        return self._standard_hts

    @property
    def check_hts(self):  # type: ignore
        """
        str: The Hilltop service to be checked.

        Setting this property will mark the data as stale.
        """
        return self._check_hts

    @property
    def quality_code_evaluator(self):  # type: ignore
        """Measurement property."""
        return self._quality_code_evaluator

    @ClassLogger  # type: ignore
    @quality_code_evaluator.setter
    def quality_code_evaluator(self, value):
        self._quality_code_evaluator = value
        self._stale = True

    @property
    def defaults(self):  # type: ignore
        """
        dict: The default settings.

        Setting this property will mark the data as stale.
        """
        return self._defaults

    @property  # type: ignore
    def standard_series(self) -> pd.Series:  # type: ignore
        """pd.Series: The standard series data."""
        return self._standard_series

    @ClassLogger  # type: ignore
    @standard_series.setter
    def standard_series(self, value):
        self._standard_series = value

    @property
    def check_data(self):  # type: ignore
        """pd.DataFrame: The DataFrame containing check data."""
        return self._check_data

    @check_data.setter
    def check_data(self, value):
        self._check_data = value
        self._check_series = self._check_data[self.check_item_name]

    @property
    def check_series(self):  # type: ignore
        """pd.Series: The series containing check data."""
        return self._check_series

    @ClassLogger  # type: ignore
    @check_series.setter
    def check_series(self, value):
        self._check_series = value
        self._check_data[self.check_item_name] = value

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
    def import_standard(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        overwrite: bool = False,
    ):
        """
        Import standard data.

        Parameters
        ----------
        from_date : str or None, optional
            The start date for data retrieval. If None, defaults to earliest available data.
        to_date : str or None, optional
            The end date for data retrieval. If None, defaults to latest available data.
        overwrite : bool, optional
            If True, overwrite existing data with newly acquired data for overlapping timestamps.
            If False, keep existing data and only add new datapoints if they don't already exist.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            - If no standard data is found within the specified date range.
            - If standard data is not found in the server response.

        TypeError
            If the parsed Standard data is not a pandas.Series.

        Warnings
        --------
        UserWarning
            - If the existing Standard Series is not a pandas.Series, it is set to an empty Series.

        Notes
        -----
        This method imports Standard data from the specified server based on the provided parameters.
        It retrieves data using the `data_acquisition.get_data` function and updates the Standard Series in the instance.
        The data is parsed and formatted according to the item_info in the data source.

        Examples
        --------
        >>> processor = Processor(...)  # initialize processor instance
        >>> processor.import_standard(from_date='2022-01-01', to_date='2022-01-10', overwrite=True)
        """
        xml_tree, blob_list = data_acquisition.get_data(
            self._base_url,
            self._standard_hts,
            self._site,
            self._standard_measurement_name,
            from_date,
            to_date,
            tstype="Standard",
        )

        if isinstance(self._standard_series, pd.Series):
            curr_series = self._standard_series
        else:
            warnings.warn(
                "Existing Standard Series should be pandas.Series, but found "
                f"{type(self._standard_series)}. Setting to empty pd.Series",
                stacklevel=1,
            )
            curr_series = pd.Series({})
        insert_series = pd.Series({})
        blob_found = False

        if blob_list is None or len(blob_list) == 0:
            raise ValueError("No standard data found within specified date range.")
        date_format = "Calendar"
        data_source_list = []
        for blob in blob_list:
            data_source_list += [blob.data_source.name]
            if (blob.data_source.name == self._standard_measurement_name) and (
                blob.data_source.ts_type == "StdSeries"
            ):
                insert_series = blob.data.timeseries
                date_format = blob.data.date_format
                if insert_series is not None:
                    # Found it. Now we extract it.
                    blob_found = True
                    self.raw_data_dict["standard"] = {
                        "data_blob": blob,
                        "raw_xml": xml_tree,
                    }
        if not blob_found:
            raise ValueError(
                f"Standard Data Not Found under name {self._standard_measurement_name}. "
                f"Available data sources are: {data_source_list}"
            )

        if not isinstance(insert_series, pd.Series):
            raise TypeError(
                f"Expecting pd.Series for Standard data, but got {type(insert_series)}"
                "from parser."
            )
        if not insert_series.empty:
            if date_format == "mowsecs":
                insert_series.index = utils.mowsecs_to_datetime_index(
                    insert_series.index
                )
            else:
                insert_series.index = pd.to_datetime(insert_series.index)
            insert_series = insert_series.asfreq(self._frequency, method="ffill")
        if not curr_series.empty:
            if overwrite:
                # Combine, but overwrite overlapping timestamps with newly acquired
                # data
                self._standard_series = insert_series.combine_first(
                    curr_series
                ).sort_index()
            else:
                # Combine, keeping all existing values and only adding new datapoints
                # if they don't already exist.
                self._standard_series = curr_series.combine_first(
                    insert_series
                ).sort_index()
        else:
            self._standard_series = insert_series

        fmt = (
            self.raw_data_dict["standard"]["data_blob"]
            .data_source.item_info[0]
            .item_format
        )
        div = (
            self.raw_data_dict["standard"]["data_blob"].data_source.item_info[0].divisor
        )
        if div is None or div == "None":
            div = 1
        if fmt == "I":
            self.standard_series = self._standard_series.astype(int) / int(div)
        elif fmt == "F":
            self.standard_series = self._standard_series.astype(np.float32) / float(div)
        elif fmt == "D":  # Not sure if this would ever really happen, but...
            self.standard_series = utils.mowsecs_to_datetime_index(
                self._standard_series
            )
        else:
            raise ValueError(f"Unknown Format Spec: {fmt}")

    @ClassLogger
    def import_quality(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        overwrite: bool = False,
    ):
        """
        Import quality data.

        Parameters
        ----------
        from_date : str or None, optional
            The start date for data retrieval. If None, defaults to earliest available data.
        to_date : str or None, optional
            The end date for data retrieval. If None, defaults to latest available data.
        overwrite : bool, optional
            If True, overwrite existing data with newly acquired data for overlapping timestamps.
            If False, keep existing data and only add new datapoints if they don't already exist.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the parsed Quality data is not a pandas.Series.

        Warnings
        --------
        UserWarning
            - If the existing Quality Series is not a pandas.Series, it is set to an empty Series.
            - If no Quality data is available for the specified date range.
            - If Quality data is not found in the server response.

        Notes
        -----
        This method imports Quality data from the specified server based on the provided parameters.
        It retrieves data using the `data_acquisition.get_data` function and updates the Quality Series in the instance.
        The data is parsed and formatted according to the item_info in the data source.

        Examples
        --------
        >>> processor = Processor(...)  # initialize processor instance
        >>> processor.import_quality(from_date='2022-01-01', to_date='2022-01-10', overwrite=True)
        """
        xml_tree, blob_list = data_acquisition.get_data(
            self._base_url,
            self._standard_hts,
            self._site,
            self._standard_measurement_name,
            from_date,
            to_date,
            tstype="Quality",
        )

        if isinstance(self._quality_series, pd.Series):
            curr_series = self._quality_series
        else:
            warnings.warn(
                "Existing Quality Series should be pandas.Series, but found "
                f"{type(self._quality_series)}. Setting to empty pd.Series",
                stacklevel=1,
            )
            curr_series = pd.Series({})
        insert_series = pd.Series({})
        blob_found = False

        if blob_list is None or len(blob_list) == 0:
            warnings.warn(
                "No Quality data available for the range specified.",
                stacklevel=2,
            )
        else:
            date_format = "Calendar"
            for blob in blob_list:
                if (blob.data_source.name == self._standard_measurement_name) and (
                    blob.data_source.ts_type == "StdQualSeries"
                ):
                    # Found it. Now we extract it.
                    blob_found = True

                    insert_series = blob.data.timeseries
                    date_format = blob.data.date_format
                    if insert_series is not None:
                        self.raw_data_dict["quality"] = {
                            "data_blob": blob,
                            "raw_xml": xml_tree,
                        }
            if not blob_found:
                warnings.warn(
                    "No Quality data found in the server response.",
                    stacklevel=2,
                )

            if not isinstance(insert_series, pd.Series):
                raise TypeError(
                    f"Expecting pd.Series for Quality data, but got {type(insert_series)}"
                    "from parser."
                )
            if not insert_series.empty:
                if date_format == "mowsecs":
                    insert_series.index = utils.mowsecs_to_datetime_index(
                        insert_series.index
                    )
                else:
                    insert_series.index = pd.to_datetime(insert_series.index)

            if not curr_series.empty:
                if overwrite:
                    # Combine, but overwrite overlapping timestamps with newly acquired
                    # data
                    self._quality_series = insert_series.combine_first(
                        curr_series
                    ).sort_index()
                else:
                    # Combine, keeping all existing values and only adding new datapoints
                    # if they don't already exist.
                    self._quality_series = curr_series.combine_first(
                        insert_series
                    ).sort_index()
            elif not insert_series.empty:
                self._quality_series = insert_series
            self.quality_series = self._quality_series.astype(int)

    @ClassLogger
    def import_check(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        overwrite: bool = False,
    ):
        """
        Import Check data.

        Parameters
        ----------
        from_date : str or None, optional
            The start date for data retrieval. If None, defaults to earliest available data.
        to_date : str or None, optional
            The end date for data retrieval. If None, defaults to latest available data.
        overwrite : bool, optional
            If True, overwrite existing data with newly acquired data for overlapping timestamps.
            If False, keep existing data and only add new datapoints if they don't already exist.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the parsed Check data is not a pandas.DataFrame.

        Warnings
        --------
        UserWarning
            - If the existing Check Data is not a pandas.DataFrame, it is set to an empty DataFrame.
            - If no Check data is available for the specified date range.
            - If the Check data source is not found in the server response.

        Notes
        -----
        This method imports Check data from the specified server based on the provided parameters.
        It retrieves data using the `data_acquisition.get_data` function and updates the Check data in the instance.
        The data is parsed and formatted according to the item_info in the data source.

        Examples
        --------
        >>> processor = Processor(...)  # initialize processor instance
        >>> processor.import_check(from_date='2022-01-01', to_date='2022-01-10', overwrite=True)
        """
        xml_tree, blob_list = data_acquisition.get_data(
            self._base_url,
            self._check_hts,
            self._site,
            self._check_measurement_name,
            from_date,
            to_date,
            tstype="Check",
        )
        if isinstance(self._check_data, pd.DataFrame):
            curr_data = self._check_data
        else:
            warnings.warn(
                "Existing Check Data should be pandas.DataFrame, but found "
                f"{type(self._check_data)}. Setting to empty pd.DataFrame",
                stacklevel=1,
            )
            curr_data = pd.DataFrame({})
        insert_data = pd.DataFrame({})
        blob_found = False

        date_format = "Calendar"
        if blob_list is None or len(blob_list) == 0:
            warnings.warn(
                "No Check data available for the range specified.",
                stacklevel=2,
            )
        else:
            data_source_options = []
            for blob in blob_list:
                data_source_options += [blob.data_source.name]
                if (blob.data_source.name == self.check_data_source_name) and (
                    blob.data_source.ts_type == "CheckSeries"
                ):
                    # Found it. Now we extract it.
                    blob_found = True

                    date_format = blob.data.date_format

                    # This could be a pd.Series
                    insert_data = blob.data.timeseries
                    if insert_data is not None:
                        self.raw_data_dict["check"] = {
                            "data_blob": blob,
                            "raw_xml": xml_tree,
                        }
            if not blob_found:
                warnings.warn(
                    f"Check data {self.check_data_source_name} not found in server "
                    f"response. Available options are {data_source_options}",
                    stacklevel=2,
                )

            if not isinstance(insert_data, pd.DataFrame):
                raise TypeError(
                    f"Expecting pd.DataFrame for Check data, but got {type(insert_data)}"
                    "from parser."
                )
            if not insert_data.empty:
                if date_format == "mowsecs":
                    insert_data.index = utils.mowsecs_to_datetime_index(
                        insert_data.index
                    )
                else:
                    insert_data.index = pd.to_datetime(insert_data.index)

            if not curr_data.empty:
                if overwrite:
                    # Combine, but overwrite overlapping timestamps with newly acquired
                    # data
                    self._check_data = insert_data.combine_first(curr_data).sort_index()
                else:
                    # Combine, keeping all existing values and only adding new datapoints
                    # if they don't already exist.
                    self._check_data = curr_data.combine_first(insert_data).sort_index()
            elif not insert_data.empty:
                self._check_data = insert_data
            if not self._check_data.empty:
                # TODO: Maybe this should happen in the parser?
                for i, item in enumerate(
                    self.raw_data_dict["check"]["data_blob"].data_source.item_info
                ):
                    fmt = item.item_format
                    div = item.divisor
                    col = self._check_data.iloc[:, i]
                    if fmt == "I":
                        self._check_data.iloc[:, i] = col.astype(int) / int(div)
                    elif fmt == "F":
                        self._check_data.iloc[:, i] = col.astype(np.float32) / float(
                            div
                        )
                    elif fmt == "D":
                        if self._check_data.iloc[:, i].dtype != pd.Timestamp:
                            if date_format == "mowsecs":
                                self._check_data.iloc[
                                    :, i
                                ] = utils.mowsecs_to_datetime_index(col)
                            else:
                                self._check_data.iloc[:, i] = col.astype(pd.Timestamp)
                    elif fmt == "S":
                        self._check_data.iloc[:, i] = col.astype(str)

            self.check_data = self._check_data
            if not self._check_data.empty:
                self.check_series = self.check_data[self.check_item_name]
            else:
                self.check_series = pd.Series({})

    def import_data(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        overwrite: bool = False,
        standard: bool = True,
        check: bool = True,
        quality: bool = True,
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
        if standard:
            self.import_standard(from_date, to_date, overwrite)
        if quality:
            self.import_quality(from_date, to_date, overwrite)
        if check:
            self.import_check(from_date, to_date, overwrite)
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
            self._check_series,
            self._quality_code_evaluator,
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
            pd.Series(self._check_series),
            low_clip,
            high_clip,
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
    def remove_flatlined_values(self, span: int = 3):
        """Remove repeated values in std series a la flatline_value_remover()."""
        self.standard_series = filters.flatline_value_remover(
            self._standard_series, span=span
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
    def data_exporter(self, file_location, type="xml", trimmed=True):
        """
        Export data to CSV file.

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
        This method exports data to a CSV file.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.data_exporter("output.csv", trimmed=True)
        >>> # Check the generated CSV file at 'output.csv'
        """
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
            self._quality_code_evaluator.name,
            std_series,
            self._check_series,
            self._quality_series,
        )
        data_sources.hilltop_export(
            file_location,
            self._site,
            self._quality_code_evaluator.name,
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

    def plot_comparison_qc_series(self, show=True):
        """Implement comparison_qc_plotter()."""
        raw_series = self.raw_data_dict["standard"]["data_blob"].data.timeseries.astype(
            float
        )
        plotter.comparison_qc_plotter(
            self._standard_series,
            raw_series,
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

        for dtype, raw in self.raw_data_dict.items():
            raw_blob = raw["data_blob"]
            if hasattr(raw_blob, "item_info") and (raw_blob.item_info) is not None:
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
                name=self._standard_measurement_name,
                num_items=raw_blob.data_source.num_items,
                ts_type=raw_blob.data_source.ts_type,
                data_type=raw_blob.data_source.data_type,
                interpolation=raw_blob.data_source.interpolation,
                item_format=raw_blob.data_source.item_format,
                item_info=raw_blob.data_source.item_info,
            )

            if dtype == "standard":
                if isinstance(self._standard_series, pd.Series):
                    timeseries = self._standard_series
                else:
                    raise TypeError(
                        "Standard Series should be pd.Series, "
                        f"found {type(self._standard_series)}"
                    )

            elif dtype == "quality":
                if isinstance(self._quality_series, pd.Series):
                    timeseries = self._quality_series
                else:
                    raise TypeError(
                        "Quality Series should be pd.Series, "
                        f"found {type(self._quality_series)}"
                    )
            elif dtype == "check":
                if isinstance(self._check_data, pd.DataFrame):
                    timeseries = self._check_data
                else:
                    raise TypeError(
                        "Check Data should be pd.DataFrame, "
                        f"found {type(self._check_data)}"
                    )
            else:
                raise ValueError("No data found for export.")
            # timeseries.index = utils.datetime_index_to_mowsecs(
            #     timeseries.index
            # )
            if (
                hasattr(raw_blob.data_source, "item_info")
                and raw_blob.data_source.item_info is not None
            ):
                for i, info in enumerate(raw_blob.data_source.item_info):
                    if info.item_format == "F":
                        pattern = re.compile(r"#+\.?(#*)")
                        match = pattern.match(info.format)
                        float_format = "{:.1f}"
                        if match:
                            group = match.group(1)
                            dp = len(group)
                            float_format = "{:." + str(dp) + "f}"
                        if isinstance(timeseries, pd.DataFrame):
                            timeseries.iloc[:, i] = timeseries.iloc[:, i].map(
                                lambda x, f=float_format: f.format(x)
                            )
                        elif isinstance(timeseries, pd.Series):
                            timeseries = timeseries.map(
                                lambda x, f=float_format: f.format(x)
                            )
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
