"""Processor class."""

import re
import warnings

import numpy as np
import pandas as pd
from annalist.annalist import Annalist
from annalist.decorators import ClassLogger
from hilltoppy import Hilltop

from hydrobot import (
    data_acquisition,
    data_sources,
    data_structure,
    evaluator,
    filters,
    plotter,
    utils,
)

annalizer = Annalist()


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
        interval_dict: dict | None = None,
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
        interval_dict : dict, optional
            Determines how data with old checks is downgraded
        kwargs : dict
            Additional keyword arguments.
        """
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

        # standard
        available_standard_measurements = standard_hilltop.get_measurement_list(site)
        self._standard_measurement_name = standard_measurement_name
        matches = re.search(r"([^\[\n]+)(\[(.+)\])?", standard_measurement_name)

        if matches is not None:
            self.standard_item_name = matches.groups()[0].strip(" ")
            self.standard_data_source_name = matches.groups()[2]
            if self.standard_data_source_name is None:
                self.standard_data_source_name = self.standard_item_name
        if standard_measurement_name not in list(
            available_standard_measurements.MeasurementName
        ):
            raise ValueError(
                f"Standard measurement name '{standard_measurement_name}' not found at"
                f" site '{site}'. "
                "Available measurements are "
                f"{list(available_standard_measurements.MeasurementName)}"
            )

        # check
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

        self.standard_item_info = {
            "ItemName": self.standard_item_name,
            "ItemFormat": "F",
            "Divisor": 1,
            "Units": "",
            "Format": "###.##",
        }

        self.check_item_info = {
            "ItemName": self.check_item_name,
            "ItemFormat": "F",
            "Divisor": 1,
            "Units": "",
            "Format": "$$$",
        }
        self._base_url = base_url
        self._standard_hts = standard_hts
        self._check_hts = check_hts
        self._frequency = frequency
        self._from_date = from_date
        self._to_date = to_date
        self._quality_code_evaluator = data_sources.get_qc_evaluator(
            standard_measurement_name
        )

        if interval_dict is None:
            self._interval_dict = {}
        else:
            self._interval_dict = interval_dict

        self._standard_data = pd.DataFrame(
            columns=[
                "Time",
                "Raw",
                "Value",
                "Changes",
                "Remove",
            ]
        ).set_index("Time")
        self._check_data = pd.DataFrame(
            columns=[
                "Time",
                "Raw",
                "Value",
                "Changes",
                "Recorder Time",
                "Comment",
                "Source",
                "QC",
            ]
        ).set_index("Time")
        self._quality_data = pd.DataFrame(
            columns=[
                "Time",
                "Raw",
                "Value",
                "Changes",
                "Reason",
            ]
        ).set_index("Time")

        self.raw_standard_blob = None
        self.raw_standard_xml = None

        self.raw_quality_blob = None
        self.raw_quality_xml = None

        self.raw_check_blob = None
        self.raw_check_xml = None

        # Load data for the first time
        self.import_data(from_date=self.from_date, to_date=self.to_date)

    @property
    def standard_measurement_name(self):  # type: ignore
        """str: The site to be processed."""
        return self._standard_measurement_name

    @property
    def site(self):  # type: ignore
        """str: The site to be processed."""
        return self._site

    @property
    def from_date(self):  # type: ignore
        """str: The start date of the data."""
        return self._from_date

    @property
    def to_date(self):  # type: ignore
        """str: The end date of the data."""
        return self._to_date

    @property
    def frequency(self):  # type: ignore
        """str: The frequency of the data."""
        return self._frequency

    @property
    def base_url(self):  # type: ignore
        """str: The base URL of the Hilltop server."""
        return self._base_url

    @property
    def standard_hts(self):  # type: ignore
        """str: The standard Hilltop service."""
        return self._standard_hts

    @property
    def check_hts(self):  # type: ignore
        """str: The Hilltop service to be checked."""
        return self._check_hts

    @property
    def quality_code_evaluator(self):  # type: ignore
        """Measurement property."""
        return self._quality_code_evaluator

    @ClassLogger
    @quality_code_evaluator.setter
    def quality_code_evaluator(self, value):
        self._quality_code_evaluator = value

    @property
    def defaults(self):  # type: ignore
        """dict: The default settings."""
        return self._defaults

    @property  # type: ignore
    def standard_data(self) -> pd.DataFrame:  # type: ignore
        """pd.Series: The standard series data."""
        return self._standard_data

    @ClassLogger  # type: ignore
    @standard_data.setter
    def standard_data(self, value):
        self._standard_data = value

    @property  # type: ignore
    def check_data(self) -> pd.DataFrame:  # type: ignore
        """pd.Series: The series containing check data."""
        return self._check_data

    @ClassLogger  # type: ignore
    @check_data.setter
    def check_data(self, value):
        self._check_data = value

    @property  # type: ignore
    def quality_data(self) -> pd.DataFrame:  # type: ignore
        """pd.Series: The quality series data."""
        return self._quality_data

    @ClassLogger  # type: ignore
    @quality_data.setter
    def quality_data(self, value):
        self._quality_data = value

    @ClassLogger
    def import_standard(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ):
        """
        Import standard data.

        Parameters
        ----------
        from_date : str or None, optional
            The start date for data retrieval. If None, defaults to earliest available
            data.
        to_date : str or None, optional
            The end date for data retrieval. If None, defaults to latest available
            data.

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
            - If the existing Standard Series is not a pandas.Series, it is set to an
            empty Series.

        Notes
        -----
        This method imports Standard data from the specified server based on the
        provided parameters.
        It retrieves data using the `data_acquisition.get_data` function and updates
        the Standard Series in the instance.
        The data is parsed and formatted according to the item_info in the data source.

        Examples
        --------
        >>> processor = Processor(...)  # initialize processor instance
        >>> processor.import_standard(
        ...     from_date='2022-01-01', to_date='2022-01-10'
        ... )
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

        blob_found = False

        date_format = "Calendar"
        data_source_list = []
        raw_standard_data = pd.DataFrame({})
        if blob_list is None or len(blob_list) == 0:
            warnings.warn(
                "No standard data found within specified date range.",
                stacklevel=1,
            )
        else:
            for blob in blob_list:
                data_source_list += [blob.data_source.name]
                if (blob.data_source.name == self.standard_data_source_name) and (
                    blob.data_source.ts_type == "StdSeries"
                ):
                    raw_standard_data = blob.data.timeseries
                    date_format = blob.data.date_format
                    if raw_standard_data is not None:
                        # Found it. Now we extract it.
                        blob_found = True
                        self.raw_standard_blob = blob
                        self.raw_standard_xml = xml_tree
                        self.standard_item_info[
                            "ItemName"
                        ] = blob.data_source.item_info[0].item_name
                        self.standard_item_info[
                            "ItemFormat"
                        ] = blob.data_source.item_info[0].item_format
                        self.standard_item_info["Divisor"] = blob.data_source.item_info[
                            0
                        ].divisor
                        self.standard_item_info["Units"] = blob.data_source.item_info[
                            0
                        ].units
                        self.standard_item_info["Format"] = blob.data_source.item_info[
                            0
                        ].format
            if not blob_found:
                raise ValueError(
                    f"Standard Data Not Found under name "
                    f"{self._standard_measurement_name}. "
                    f"Available data sources are: {data_source_list}"
                )

            if not isinstance(raw_standard_data, pd.DataFrame):
                raise TypeError(
                    "Expecting pd.DataFrame for Standard data, "
                    f"but got {type(raw_standard_data)} from parser."
                )
            if not raw_standard_data.empty:
                if date_format == "mowsecs":
                    raw_standard_data.index = utils.mowsecs_to_datetime_index(
                        raw_standard_data.index
                    )
                else:
                    raw_standard_data.index = pd.to_datetime(raw_standard_data.index)
                raw_standard_data = raw_standard_data.asfreq(
                    self._frequency, fill_value=np.NaN
                )
            if self.raw_standard_blob is not None:
                fmt = self.standard_item_info["ItemFormat"]
                div = self.standard_item_info["Divisor"]
            else:
                warnings.warn(
                    "Could not extract standard data format from data source. "
                    "Defaulting to float format.",
                    stacklevel=1,
                )
                fmt = "F"
                div = 1
            if div is None or div == "None":
                div = 1
            if fmt == "I":
                raw_standard_data.iloc[:, 0] = raw_standard_data.iloc[:, 0].astype(
                    int
                ) / int(div)
            elif fmt == "F":
                raw_standard_data.iloc[:, 0] = raw_standard_data.iloc[:, 0].astype(
                    np.float32
                ) / float(div)
            elif fmt == "D":  # Not sure if this would ever really happen, but...
                raw_standard_data.iloc[:, 0] = utils.mowsecs_to_datetime_index(
                    raw_standard_data.iloc[:, 0]
                )
            else:
                raise ValueError(f"Unknown Format Spec: {fmt}")

            self.standard_data["Raw"] = raw_standard_data.iloc[:, 0]
            self.standard_data["Value"] = self.standard_data["Raw"]

    @ClassLogger
    def import_quality(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ):
        """
        Import quality data.

        Parameters
        ----------
        from_date : str or None, optional
            The start date for data retrieval. If None, defaults to earliest available
            data.
        to_date : str or None, optional
            The end date for data retrieval. If None, defaults to latest available
            data.

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
            - If the existing Quality Series is not a pandas.Series, it is set to an
                empty Series.
            - If no Quality data is available for the specified date range.
            - If Quality data is not found in the server response.

        Notes
        -----
        This method imports Quality data from the specified server based on the
        provided parameters. It retrieves data using the `data_acquisition.get_data`
        function and updates the Quality Series in the instance. The data is parsed and
        formatted according to the item_info in the data source.

        Examples
        --------
        >>> processor = Processor(...)  # initialize processor instance
        >>> processor.import_quality(
        ...     from_date='2022-01-01', to_date='2022-01-10', overwrite=True
        ... )
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

        blob_found = False
        raw_quality_data = pd.DataFrame({})

        if blob_list is None or len(blob_list) == 0:
            warnings.warn(
                "No Quality data available for the range specified.",
                stacklevel=1,
            )
        else:
            date_format = "Calendar"
            for blob in blob_list:
                if (blob.data_source.name == self.standard_data_source_name) and (
                    blob.data_source.ts_type == "StdQualSeries"
                ):
                    # Found it. Now we extract it.
                    blob_found = True

                    raw_quality_data = blob.data.timeseries
                    date_format = blob.data.date_format
                    if raw_quality_data is not None:
                        # Found it. Now we extract it.
                        blob_found = True
                        self.raw_quality_blob = blob
                        self.raw_quality_xml = xml_tree
            if not blob_found:
                warnings.warn(
                    "No Quality data found in the server response.",
                    stacklevel=2,
                )

            if not isinstance(raw_quality_data, pd.DataFrame):
                raise TypeError(
                    f"Expecting pd.DataFrame for Quality data, but got "
                    f"{type(raw_quality_data)} from parser."
                )
            if not raw_quality_data.empty:
                if date_format == "mowsecs":
                    raw_quality_data.index = utils.mowsecs_to_datetime_index(
                        raw_quality_data.index
                    )
                else:
                    raw_quality_data.index = pd.to_datetime(raw_quality_data.index)

            raw_quality_data.iloc[:, 0] = raw_quality_data.iloc[:, 0].astype(int)

            self.quality_data["Raw"] = raw_quality_data.iloc[:, 0]
            self.quality_data["Value"] = self.quality_data["Raw"]

    @ClassLogger
    def import_check(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ):
        """
        Import Check data.

        Parameters
        ----------
        from_date : str or None, optional
            The start date for data retrieval. If None, defaults to earliest available
            data.
        to_date : str or None, optional
            The end date for data retrieval. If None, defaults to latest available
            data.

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
            - If the existing Check Data is not a pandas.DataFrame, it is set to an
                empty DataFrame.
            - If no Check data is available for the specified date range.
            - If the Check data source is not found in the server response.

        Notes
        -----
        This method imports Check data from the specified server based on the provided
        parameters. It retrieves data using the `data_acquisition.get_data` function.
        The data is parsed and formatted according to the item_info in the data source.

        Examples
        --------
        >>> processor = Processor(...)  # initialize processor instance
        >>> processor.import_check(
        ...     from_date='2022-01-01', to_date='2022-01-10', overwrite=True
        ... )
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
        import_data = pd.DataFrame({})
        blob_found = False

        date_format = "Calendar"
        if blob_list is None or len(blob_list) == 0:
            warnings.warn(
                "No Check data available for the range specified.",
                stacklevel=2,
            )
        else:
            raw_check_data = pd.DataFrame({})
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
                    import_data = blob.data.timeseries
                    if import_data is not None:
                        self.raw_check_blob = blob
                        self.raw_check_xml = xml_tree
                        raw_check_data = import_data
                        self.check_item_info["ItemName"] = blob.data_source.item_info[
                            0
                        ].item_name
                        self.check_item_info["ItemFormat"] = blob.data_source.item_info[
                            0
                        ].item_format
                        self.check_item_info["Divisor"] = blob.data_source.item_info[
                            0
                        ].divisor
                        self.check_item_info["Units"] = blob.data_source.item_info[
                            0
                        ].units
                        self.check_item_info["Format"] = blob.data_source.item_info[
                            0
                        ].format
            if not blob_found:
                warnings.warn(
                    f"Check data {self.check_data_source_name} not found in server "
                    f"response. Available options are {data_source_options}",
                    stacklevel=2,
                )

            if not isinstance(raw_check_data, pd.DataFrame):
                raise TypeError(
                    f"Expecting pd.DataFrame for Check data, but got {type(raw_check_data)}"
                    "from parser."
                )
            if not raw_check_data.empty:
                if date_format == "mowsecs":
                    raw_check_data.index = utils.mowsecs_to_datetime_index(
                        raw_check_data.index
                    )
                else:
                    raw_check_data.index = pd.to_datetime(raw_check_data.index)

            if not raw_check_data.empty and self.raw_check_blob is not None:
                # TODO: Maybe this should happen in the parser?
                for i, item in enumerate(self.raw_check_blob.data_source.item_info):
                    fmt = item.item_format
                    div = item.divisor
                    col = raw_check_data.iloc[:, i]
                    if fmt == "I":
                        raw_check_data.iloc[:, i] = col.astype(int) / int(div)
                    elif fmt == "F":
                        raw_check_data.iloc[:, i] = col.astype(np.float32) / float(div)
                    elif fmt == "D":
                        if raw_check_data.iloc[:, i].dtype != pd.Timestamp:
                            if date_format == "mowsecs":
                                raw_check_data.iloc[
                                    :, i
                                ] = utils.mowsecs_to_datetime_index(col)
                            else:
                                raw_check_data.iloc[:, i] = col.astype(pd.Timestamp)
                    elif fmt == "S":
                        raw_check_data.iloc[:, i] = col.astype(str)

            if not raw_check_data.empty:
                self.check_data["Raw"] = raw_check_data[self.check_item_name]
                self.check_data["Value"] = self.check_data["Raw"]
                self.check_data["Recorder Time"] = raw_check_data["Recorder Time"]
                self.check_data["Comment"] = raw_check_data["Comment"]
                self.check_data["Source"] = "HTP"
                self.check_data["QC"] = True

    def import_data(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
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
        False
        """
        if standard:
            self.import_standard(from_date, to_date)
        if quality:
            self.import_quality(from_date, to_date)
        if check:
            self.import_check(from_date, to_date)

    @ClassLogger
    def add_standard(self, extra_standard):
        """
        Incorporate extra standard data into the standard series using utils.merge_series.

        Parameters
        ----------
        extra_standard
            extra standard data

        Returns
        -------
        None, but adds data to self.standard_series
        """
        combined = utils.merge_series(self.standard_series, extra_standard)
        self.standard_series = combined

    @ClassLogger
    def add_check(self, extra_check):
        """
        Incorporate extra check data into the check series using utils.merge_series.

        Parameters
        ----------
        extra_check
            extra check data

        Returns
        -------
        None, but adds data to self.check_series
        """
        combined = utils.merge_series(self.check_series, extra_check)
        self.check_series = combined

    @ClassLogger
    def add_quality(self, extra_quality):
        """
        Incorporate extra quality data into the quality series using utils.merge_series.

        Parameters
        ----------
        extra_quality
            extra quality data

        Returns
        -------
        None, but adds data to self.quality_series
        """
        combined = utils.merge_series(self.quality_series, extra_quality)
        self.quality_series = combined

    @ClassLogger
    def gap_closer(self, gap_limit: int | None = None):
        """
        Close small gaps in the standard series.

        DEPRECATED: The use of this method is discouraged as it completely removes rows
        from the dataframes. The gap closing functionality has been moved to
        data_exporter, where gaps are handled automatically before data export.

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
        warnings.warn(
            "DEPRECATED: The use of gap_closer is discouraged as it completely "
            "removes rows from the dataframes.",
            stacklevel=1,
        )
        if gap_limit is None:
            if "gap_limit" not in self._defaults:
                raise ValueError("gap_limit value required, no value found in defaults")
            else:
                gap_limit = int(self._defaults["gap_limit"])

        gapless = evaluator.small_gap_closer(
            self._standard_data["Value"].squeeze(), gap_limit=gap_limit
        )
        self._standard_data = self._standard_data.loc[gapless.index]

    @ClassLogger
    def quality_encoder(
        self,
        gap_limit: int | None = None,
        max_qc: int | float | None = None,
        interval_dict: dict | None = None,
    ):
        """
        Encode quality information in the quality series.

        Parameters
        ----------
        gap_limit : int or None, optional
            The maximum number of consecutive missing values to consider as gaps, by
            default None.
            If None, the gap limit from the class defaults is used.
        max_qc : numeric or None, optional
            Maximum quality code possible at site
            If None, the max qc from the class defaults is used.
        interval_dict : dict or None, optional
            Dictionary that dictates when to downgrade data with old checks
            Takes pd.DateOffset:quality_code pairs
            If None, the interval_dict from the class defaults is used.

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
            if "gap_limit" not in self._defaults:
                raise ValueError("gap_limit value required, no value found in defaults")
            else:
                gap_limit = int(self._defaults["gap_limit"])
        if max_qc is None:
            max_qc = self._defaults["max_qc"] if "max_qc" in self._defaults else np.NaN

        if interval_dict is None:
            interval_dict = self._interval_dict

        qc_checks = self.check_data[self.check_data["QC"]]
        self.quality_data["Value"] = evaluator.quality_encoder(
            self._standard_data["Value"],
            qc_checks["Value"],
            self._quality_code_evaluator,
            gap_limit=gap_limit,
            max_qc=max_qc,
            interval_dict=interval_dict,
        )

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
            low_clip = (
                float(self._defaults["low_clip"])
                if "low_clip" in self._defaults
                else np.NaN
            )
        if high_clip is None:
            high_clip = (
                float(self._defaults["high_clip"])
                if "high_clip" in self._defaults
                else np.NaN
            )

        clipped = filters.clip(
            self._standard_data["Value"].squeeze(), low_clip, high_clip
        )
        self._standard_data = self._apply_changes(
            self._standard_data, clipped, "CLP", mark_remove=True
        )

    @staticmethod
    def _apply_changes(
        dataframe,
        changed_values,
        change_code,
        mark_remove=False,
    ):
        diffs = dataframe["Value"] != changed_values
        if mark_remove:
            dataframe.loc[diffs, "Remove"] = mark_remove
        dataframe.loc[diffs, "Changes"] = change_code
        dataframe["Value"] = changed_values
        return dataframe

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
            if "span" not in self._defaults:
                raise ValueError("span value required, no value found in defaults")
            else:
                span = int(self._defaults["span"])
        if delta is None:
            if "delta" not in self._defaults:
                raise ValueError("delta value required, no value found in defaults")
            else:
                delta = float(self._defaults["delta"])

        rm_outliers = filters.remove_outliers(
            self._standard_data["Value"].squeeze(), span, delta
        )

        self._standard_data = self._apply_changes(
            self._standard_data, rm_outliers, "OUT", mark_remove=True
        )

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
            low_clip = (
                float(self._defaults["low_clip"])
                if "low_clip" in self._defaults
                else np.NaN
            )
        if high_clip is None:
            high_clip = (
                float(self._defaults["high_clip"])
                if "low_clip" in self._defaults
                else np.NaN
            )
        if span is None:
            if "span" not in self._defaults:
                raise ValueError("span value required, no value found in defaults")
            else:
                span = int(self._defaults["span"])
        if delta is None:
            if "delta" not in self._defaults:
                raise ValueError("delta value required, no value found in defaults")
            else:
                delta = float(self._defaults["delta"])

        rm_spikes = filters.remove_spikes(
            self._standard_data["Value"].squeeze(),
            span,
            low_clip,
            high_clip,
            delta,
        )

        self._standard_data = self._apply_changes(
            self._standard_data, rm_spikes, "SPK", mark_remove=True
        )

    @ClassLogger
    def remove_flatlined_values(self, span: int = 3):
        """Remove repeated values in std series a la flatline_value_remover()."""
        rm_fln = filters.flatline_value_remover(self._standard_data["Value"], span=span)

        self._standard_data = self._apply_changes(
            self._standard_data, rm_fln, "FLN", mark_remove=True
        )

    @ClassLogger
    def remove_range(
        self,
        from_date,
        to_date,
    ):
        """
        Mark a range in standard_data for removal.

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
        >>> processor.remove_range(from_date="2022-01-01", to_date="2022-12-31", \
                tstype_standard=True)
        >>> processor.standard_data
        <standard series with specified range deleted>
        >>> processor.remove_range(from_date="2022-01-01", to_date="2022-12-31", \
                tstype_check=True)
        >>> processor.check_data
        <check series with specified range deleted>
        """
        rm_range = filters.remove_range(
            self._standard_data["Value"],
            from_date,
            to_date,
            insert_gaps="all",
        )
        self.standard_data = self._apply_changes(
            self._standard_data, rm_range, "MAN", mark_remove=True
        )

    @ClassLogger
    def delete_range(
        self,
        from_date,
        to_date,
        tstype_standard=True,
        tstype_check=False,
        tstype_quality=False,
        gap_limit=None,
    ):
        """
        Delete a range of data from specified time series types.

        DEPRECATED: The use of this method is discouraged as it completely removes rows
        from the dataframes. User is encouraged to use 'remove_range' which marks rows
        for removal, but retains the timestamp to be associated with the other values
        in the row such as the raw value, reason for removal, etc.

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
        >>> processor.standard_data
        <standard series with specified range deleted>
        >>> processor.delete_range(from_date="2022-01-01", to_date="2022-12-31", \
                tstype_check=True)
        >>> processor.check_data
        <check series with specified range deleted>
        """
        warnings.warn(
            "DEPRECATED: The use of delete_range is discouraged as it completely "
            "removes rows from the dataframes. User is encouraged to use "
            "'remove_range' which marks rows for removal, but retains the timestamp "
            "to be associated with the other values "
            "in the row such as the raw value, reason for removal, etc.",
            stacklevel=1,
        )
        if gap_limit is None:
            if "gap_limit" not in self._defaults:
                raise ValueError("gap_limit value required, no value found in defaults")
            else:
                gap_limit = self._defaults["gap_limit"]

        if tstype_standard:
            self.standard_data = filters.remove_range(
                self._standard_data,
                from_date,
                to_date,
                min_gap_length=gap_limit,
                insert_gaps="start",
            )
        if tstype_check:
            self.check_data = filters.remove_range(
                self._check_data,
                from_date,
                to_date,
                min_gap_length=gap_limit,
                insert_gaps="start",
            )
        if tstype_quality:
            self.quality_data = filters.remove_range(
                self._quality_data,
                from_date,
                to_date,
                min_gap_length=gap_limit,
                insert_gaps="start",
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
        self.standard_data = self._standard_data.asfreq(self._frequency)

    @ClassLogger
    def data_exporter(
        self,
        file_location,
        ftype="xml",
        standard: bool = True,
        quality: bool = True,
        check: bool = True,
        trimmed=True,
    ):
        """
        Export data to CSV file.

        Parameters
        ----------
        file_location : str
            The file path where the file will be saved. If 'ftype' is "csv" or "xml",
            this should be a full file path including extension. If 'ftype' is
            "hilltop_csv", multiple files will be created, so 'file_location' should be
            a prefix that will be appended with "_std_qc.csv" for the file containing
            the standard and quality data, and "_check.csv" for the check data file.
        ftype : str, optional
            Avalable options are "xml", "hilltop_csv", "csv", "check".
        trimmed : bool, optional
            If True, export trimmed data; otherwise, export the full data.
            Default is True.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            - If ftype is not a recognised string

        Notes
        -----
        This method exports data to a CSV file.

        Examples
        --------
        >>> processor = Processor(base_url="https://hilltop-server.com", site="Site1")
        >>> processor.data_exporter("output.xml", trimmed=True)
        >>> # Check the generated XML file at 'output.xml'
        """
        export_selections = [standard, quality, check]
        if trimmed:
            std_data = filters.trim_series(
                self._standard_data["Value"],
                self._check_data["Value"],
            )
        else:
            std_data = self._standard_data

        if ftype == "csv":
            all_data = [
                self._standard_data["Value"],
                self._quality_data["Value"],
                self._check_data["Value"],
            ]
            columns = ["Standard", "Quality", "Check"]

            for data, col in zip(all_data, columns, strict=True):
                data.name = col

            export_list = [
                i for (i, v) in zip(all_data, export_selections, strict=True) if v
            ]
            data_sources.series_export_to_csv(file_location, series=export_list)
        elif ftype == "hilltop_csv":
            data_sources.hilltop_export(
                file_location,
                self._site,
                self._quality_code_evaluator.name,
                std_data,
                self._check_data["Value"],
                self._quality_data["Value"],
            )
        elif ftype == "xml":
            blob_list = self.to_xml_data_structure(
                standard=standard, quality=quality, check=check
            )
            data_structure.write_hilltop_xml(blob_list, file_location)
        else:
            raise ValueError("Invalid ftype (filetype)")

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
            self._standard_data["Value"],
            self._check_data["Value"],
            self._quality_data["Value"],
            self._frequency,
        )

    def plot_qc_series(self, check=False, show=True):
        """Implement qc_plotter()."""
        fig = plotter.qc_plotter_plotly(
            self._standard_data["Value"],
            (self._check_data["Value"] if check else None),
            self._quality_data["Value"],
            self._frequency,
            show=show,
        )
        return fig

    def plot_comparison_qc_series(self, show=True):
        """Implement comparison_qc_plotter()."""
        fig = plotter.comparison_qc_plotter_plotly(
            self._standard_data["Value"],
            self._standard_data["Raw"],
            self._check_data["Value"],
            self._quality_data["Value"],
            self._frequency,
            show=show,
        )
        return fig

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
            plotter.gap_plotter(self._standard_data["Value"], show=show)
        else:
            plotter.gap_plotter(self._standard_data["Value"], span, show=show)

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
            plotter.check_plotter(
                self._standard_data["Value"],
                self._check_data["Value"],
                show=show,
            )
        else:
            plotter.check_plotter(
                self._standard_data["Value"],
                self._check_data["Value"],
                span,
                show=show,
            )

    def to_xml_data_structure(self, standard=True, quality=True, check=True):
        """
        Convert Processor object data to a list of XML data structures.

        Returns
        -------
        list of data_structure.DataSourceBlob
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

        if standard:
            standard_item_info = data_structure.ItemInfo(
                item_number=1,
                item_name=self.standard_item_info["ItemName"],
                item_format=self.standard_item_info["ItemFormat"],
                divisor=self.standard_item_info["Divisor"],
                units=self.standard_item_info["Units"],
                format=self.standard_item_info["Format"],
            )
            standard_data_source = data_structure.DataSource(
                name=self.standard_data_source_name,
                num_items=1,
                ts_type="StdSeries",
                data_type="SimpleTimeSeries",
                interpolation="Instant",
                item_format="1",
                item_info=[standard_item_info],
            )
            formatted_std_timeseries = self.standard_data["Value"].astype(str)
            if standard_item_info.item_format == "F":
                pattern = re.compile(r"#+\.?(#*)")
                match = pattern.match(standard_item_info.format)
                float_format = "{:.1f}"
                if match:
                    group = match.group(1)
                    dp = len(group)
                    float_format = "{:." + str(dp) + "f}"
                    formatted_std_timeseries = (
                        self.standard_data["Value"]
                        .astype(np.float64)
                        .map(lambda x, f=float_format: f.format(x))
                    )

            actual_nan_timeseries = formatted_std_timeseries.replace("nan", np.nan)

            # TODO: Handle gaps
            if "gap_limit" not in self._defaults:
                pass
            else:
                standard_timeseries = evaluator.small_gap_closer(
                    actual_nan_timeseries,
                    gap_limit=self._defaults["gap_limit"],
                )

            standard_data = data_structure.Data(
                date_format="Calendar",
                num_items=3,
                timeseries=standard_timeseries.to_frame(),
            )

            standard_data_blob = data_structure.DataSourceBlob(
                site_name=self.site,
                data_source=standard_data_source,
                data=standard_data,
            )
            data_blob_list += [standard_data_blob]

        if check:
            check_item_info = data_structure.ItemInfo(
                item_number=1,
                item_name=self.check_item_info["ItemName"],
                item_format=self.check_item_info["ItemFormat"],
                divisor=self.check_item_info["Divisor"],
                units=self.check_item_info["Units"],
                format=self.check_item_info["Format"],
            )
            recorder_time_item_info = data_structure.ItemInfo(
                item_number=2,
                item_name="Recorder Time",
                item_format="D",
                divisor="1",
                units="",
                format="###",
            )
            comment_item_info = data_structure.ItemInfo(
                item_number=3,
                item_name="Comment",
                item_format="F",
                divisor="1",
                units="",
                format="###",
            )

            check_data_source = data_structure.DataSource(
                name=self.check_data_source_name,
                num_items=3,
                ts_type="CheckSeries",
                data_type="SimpleTimeSeries",
                interpolation="Discrete",
                item_format="45",
                item_info=[
                    check_item_info,
                    recorder_time_item_info,
                    comment_item_info,
                ],
            )

            if check_item_info.item_format == "F":
                pattern = re.compile(r"#+\.?(#*)")
                match = pattern.match(check_item_info.format)
                float_format = "{:.1f}"
                if match:
                    group = match.group(1)
                    dp = len(group)
                    float_format = "{:." + str(dp) + "f}"
                    self.check_data.loc[:, "Value"] = self.check_data.loc[
                        :, "Value"
                    ].map(lambda x, f=float_format: f.format(x))

            check_data = data_structure.Data(
                date_format="Calendar",
                num_items=3,
                timeseries=self.check_data[["Value", "Recorder Time", "Comment"]],
            )

            check_data_blob = data_structure.DataSourceBlob(
                site_name=self.site,
                data_source=check_data_source,
                data=check_data,
            )
            data_blob_list += [check_data_blob]

        if quality:
            quality_data_source = data_structure.DataSource(
                name=self.standard_data_source_name,
                num_items=1,
                ts_type="StdQualSeries",
                data_type="SimpleTimeSeries",
                interpolation="Event",
                item_format="0",
            )

            quality_data = data_structure.Data(
                date_format="Calendar",
                num_items=3,
                timeseries=self.quality_data["Value"].to_frame(),
            )

            quality_data_blob = data_structure.DataSourceBlob(
                site_name=self.site,
                data_source=quality_data_source,
                data=quality_data,
            )
            data_blob_list += [quality_data_blob]
        return data_blob_list


def hydrobot_config_yaml_init(config_path):
    """
    Initialises a Processor class given a config file.

    Parameters
    ----------
    config_path : string
        Path to config.yaml.

    Returns
    -------
    Processor, Annalist
    """
    processing_parameters = data_acquisition.config_yaml_import(config_path)

    #######################################################################################
    # Setting up logging with Annalist
    #######################################################################################

    ann = Annalist()
    ann.configure(
        logfile=processing_parameters["logfile"],
        analyst_name=processing_parameters["analyst_name"],
        stream_format_str=processing_parameters["format"]["stream"],
        file_format_str=processing_parameters["format"]["file"],
    )

    #######################################################################################
    # Creating a Hydrobot Processor object which contains the data to be processed
    #######################################################################################

    data = Processor(
        processing_parameters["base_url"],
        processing_parameters["site"],
        processing_parameters["standard_hts_filename"],
        processing_parameters["standard_measurement_name"],
        processing_parameters["frequency"],
        processing_parameters["from_date"],
        processing_parameters["to_date"],
        processing_parameters["check_hts_filename"],
        processing_parameters["check_measurement_name"],
        processing_parameters["defaults"],
        processing_parameters["inspection_expiry"],
    )
    return data, ann
