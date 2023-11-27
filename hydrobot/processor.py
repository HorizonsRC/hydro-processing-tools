"""Processor class."""

from annalist.annalist import Annalist
from hilltoppy import Hilltop
import warnings
import pandas as pd
from hydrobot import filters, data_acquisition, evaluator, data_sources, plotter
from functools import wraps

annalizer = Annalist()


def stale_warning(method):
    """
    Decorate dangerous functions.

    Check whether the data is stale, and warn user if so.
    Warning will then take input form user to determine whether to proceed or cancel.
    Cancelling will return a null function, which returns None with no side effects no matter what the input

    Parameters
    ----------
    method : function
        A function that might have some problems if the parameters have been changed but the data hasn't been
        updated

    Returns
    -------
    function
        null function if warning is heeded, otherwise
    """

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        if self._stale:
            warnings.warn(
                "Warning: a key parameter of the data has changed but the data itself has not been reloaded."
            )
            while True:
                user_input = input("Do you want to continue? y/n: ")

                if user_input.lower() in ["y", "ye", "yes"]:
                    print("Continuing")
                    return method(self, *method_args, **method_kwargs)
                elif user_input.lower() in ["n", "no"]:
                    print("Function cancelled")
                    return lambda *x: None
                else:
                    print(
                        "Type y or n (or yes or no, or even ye, all ye who enter here)"
                    )
        else:
            return method(self, *method_args, **method_kwargs)

    return _impl


class Processor:
    """docstring for Processor."""

    @annalizer.annalize
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
        defaults: dict = {},
        **kwargs,
    ):
        """Initialize a Processor instance."""
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
                f"Available sites in standard_hts are {[s for s in standard_hilltop.available_sites]}"
                f"Available sites in check_hts are {[s for s in check_hilltop.available_sites]}"
            )

        self._standard_measurement_list = standard_hilltop.get_measurement_list(site)
        if standard_measurement in self._standard_measurement_list.values:
            self._standard_measurement = standard_measurement
        else:
            raise ValueError(
                f"Standard measurement '{standard_measurement}' not found at site '{site}'. "
                "Available measurements are "
                f"{[str(m[0]) for m in self._standard_measurement_list.values]}"
            )
        self._check_measurement_list = check_hilltop.get_measurement_list(site)
        if check_measurement in self._check_measurement_list.values:
            self._check_measurement = check_measurement
        else:
            raise ValueError(
                f"Check measurement '{check_measurement}' not found at site '{site}'. "
                "Available measurements are "
                f"{[str(m[0]) for m in self._check_measurement_list.values]}"
            )

        self._base_url = base_url
        self._standard_hts = standard_hts
        self._check_hts = check_hts
        self._frequency = frequency
        self._from_date = from_date
        self._to_date = to_date
        self._defaults = defaults
        self._measurement = data_sources.get_measurement(standard_measurement)

        self._stale = True
        self._standard_series = None
        self._check_series = None
        self._qc_series = None

        # Load data for the first time
        self.import_data()

    @property
    def from_date(self):
        """The from_date property."""
        return self._from_date

    @from_date.setter
    @annalizer.annalize
    def from_date(self, value):
        self._from_date = value
        self._stale = True

    @property
    def to_date(self):
        """The to_date property."""
        return self._to_date

    @to_date.setter
    @annalizer.annalize
    def to_date(self, value):
        self._to_date = value
        self._stale = True

    @property
    def frequency(self):
        """The to_date property."""
        return self._frequency

    @frequency.setter
    @annalizer.annalize
    def frequency(self, value):
        self._frequency = value
        self._stale = True

    @property
    def standard_series(self):
        """The dataset property."""
        return self._standard_series

    @standard_series.setter
    @annalizer.annalize
    def standard_series(self, value):
        self._standard_series = value

    @property
    def check_series(self):
        """The dataset property."""
        return self._check_series

    @check_series.setter
    @annalizer.annalize
    def check_series(self, value):
        self._check_series = value

    @annalizer.annalize
    def import_data(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        standard: bool = True,
        check: bool = True,
        quality: bool = False,
    ):
        """(Re)Load Raw Data from Hilltop."""
        if from_date is None:
            from_date = self._from_date
        if to_date is None:
            to_date = self._to_date

        if standard:
            self._standard_series = data_acquisition.get_series(
                self._base_url,
                self._standard_hts,
                self._site,
                self._standard_measurement,
                from_date,
                to_date,
                tstype="Standard",
            )
            self._standard_series = self._standard_series.asfreq(self._frequency)
        if check:
            self._check_series = data_acquisition.get_series(
                self._base_url,
                self._check_hts,
                self._site,
                self._check_measurement,
                from_date,
                to_date,
                tstype="Check",
            )
        if quality:
            self._qc_series = data_acquisition.get_series(
                self._base_url,
                self._standard_hts,
                self._site,
                self._standard_measurement,
                from_date,
                to_date,
                tstype="Quality",
            )
        self._stale = False

    @stale_warning
    def gap_closer(self, gap_limit: int | None = None):
        """Gap closer implementation."""
        if gap_limit is None:
            gap_limit = self._defaults["gap_limit"]
        self._standard_series = evaluator.small_gap_closer(
            self._standard_series, gap_limit=gap_limit
        )

    @stale_warning
    def quality_encoder(self, gap_limit: int | None = None):
        """Gap closer implementation."""
        if gap_limit is None:
            gap_limit = self._defaults["gap_limit"]
        self._qc_series = evaluator.quality_encoder(
            self._standard_series,
            self._check_series,
            self._measurement,
            gap_limit=gap_limit,
        )

    @stale_warning
    @annalizer.annalize
    def clip(self, low_clip: float | None = None, high_clip: float | None = None):
        """Clip data.

        Method implementation of filters.clip
        """
        if low_clip is None:
            low_clip = self._defaults["low_clip"]
        if high_clip is None:
            high_clip = self._defaults["high_clip"]

        self._standard_series = filters.clip(self._standard_series, low_clip, high_clip)
        self._check_series = filters.clip(self._check_series, low_clip, high_clip)

    @stale_warning
    @annalizer.annalize
    def remove_outliers(self, span: int | None = None, delta: float | None = None):
        """Remove Outliers.

        Method implementation of filters.remove_outliers
        """
        if span is None:
            span = self._defaults["span"]
        if delta is None:
            delta = self._defaults["delta"]

        self._standard_series = filters.remove_outliers(
            self._standard_series, span, delta
        )

    @stale_warning
    @annalizer.annalize
    def remove_spikes(
        self,
        low_clip: float | None = None,
        high_clip: float | None = None,
        span: int | None = None,
        delta: float | None = None,
    ):
        """Remove Spikes.

        Method implementation of filters.remove_spikes
        """
        if low_clip is None:
            low_clip = self._defaults["low_clip"]
        if high_clip is None:
            high_clip = self._defaults["high_clip"]
        if span is None:
            span = self._defaults["span"]
        if delta is None:
            delta = self._defaults["delta"]
        self._standard_series = filters.remove_spikes(
            self._standard_series, span, low_clip, high_clip, delta
        )

    def data_exporter(self, file_location):
        """Export data to csv."""
        data_sources.series_export_to_csv(
            file_location,
            self._site,
            self._measurement.name,
            self._standard_series,
            self._check_series,
            self._qc_series,
        )

    def diagnosis(self):
        """Describe the state of the data."""
        evaluator.diagnose_data(
            self._standard_series,
            self._check_series,
            self._qc_series,
            self._frequency,
        )

    def plot_qc_series(self):
        """Implement qc_plotter()."""
        plotter.qc_plotter(
            self._standard_series, self._check_series, self._qc_series, self._frequency
        )

    def plot_gaps(self, span=None):
        """Implement gap_plotter()."""
        if span is None:
            plotter.gap_plotter(self._standard_series)
        else:
            plotter.gap_plotter(self._standard_series, span)

    def plot_checks(self, span=None):
        """Implement check_plotter()."""
        if span is None:
            plotter.check_plotter(self._standard_series, self._check_series)
        else:
            plotter.check_plotter(self._standard_series, self._check_series, span)
