"""Dissolved Oxygen Processor Class."""

import re
from datetime import datetime

from annalist.annalist import Annalist
from annalist.decorators import ClassLogger
from hilltoppy import Hilltop

from hydrobot.data_acquisition import config_yaml_import
from hydrobot.evaluator import cap_qc_where_std_high
from hydrobot.processor import (
    EMPTY_QUALITY_DATA,
    EMPTY_STANDARD_DATA,
    Processor,
)
from hydrobot.utils import compare_two_qc_take_min, correct_dissolved_oxygen


class DOProcessor(Processor):
    """Processor class specifically for Dissolved Oxygen."""

    def __init__(
        self,
        base_url: str,
        site: str,
        standard_hts: str,
        standard_measurement_name: str,
        frequency: str,
        site_altitude: float,
        water_temperature_site: str,
        atmospheric_pressure_site: str,
        water_temperature_hts: str,
        atmospheric_pressure_hts: str,
        atmospheric_pressure_frequency: str,
        water_temperature_frequency: str,
        atmospheric_pressure_site_altitude: float,
        water_temperature_measurement_name: str = "Water Temperature",
        atmospheric_pressure_measurement_name: str = "Atmospheric Pressure",
        from_date: str | None = None,
        to_date: str | None = None,
        check_hts: str | None = None,
        check_measurement_name: str | None = None,
        defaults: dict | None = None,
        interval_dict: dict | None = None,
        constant_check_shift: float = 0,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            site=site,
            standard_hts=standard_hts,
            standard_measurement_name=standard_measurement_name,
            frequency=frequency,
            from_date=from_date,
            to_date=to_date,
            check_hts=check_hts,
            check_measurement_name=check_measurement_name,
            defaults=defaults,
            interval_dict=interval_dict,
            constant_check_shift=constant_check_shift,
            **kwargs,
        )

        wt_hilltop = Hilltop(base_url, water_temperature_hts, **kwargs)
        ap_hilltop = Hilltop(base_url, atmospheric_pressure_hts, **kwargs)

        if water_temperature_site is None:
            self.water_temperature_site = self.site
        else:
            self.water_temperature_site = water_temperature_site
        if atmospheric_pressure_site is None:
            self.atmospheric_pressure_site = self.site
        else:
            self.atmospheric_pressure_site = atmospheric_pressure_site

        if self.water_temperature_site not in wt_hilltop.available_sites:
            raise ValueError(
                f"Water Temperature site '{self.water_temperature_site}' not found for both base_url and hts combos."
                f"Available sites in {water_temperature_hts} are: "
                f"{[s for s in wt_hilltop.available_sites]}"
            )

        if self.atmospheric_pressure_site not in ap_hilltop.available_sites:
            raise ValueError(
                f"Atmospheric Pressure site '{site}' not found for both base_url and hts combos."
                f"Available sites in {atmospheric_pressure_hts} are: "
                f"{[s for s in ap_hilltop.available_sites]}"
            )

        # Atmospheric Pressure
        available_ap_measurements = ap_hilltop.get_measurement_list(site)
        self.atmospheric_pressure_measurement_name = (
            atmospheric_pressure_measurement_name
        )
        matches = re.search(
            r"([^\[\n]+)(\[(.+)\])?", atmospheric_pressure_measurement_name
        )

        if matches is not None:
            self.ap_item_name = matches.groups()[0].strip(" ")
            self.ap_data_source_name = matches.groups()[2]
            if self.ap_data_source_name is None:
                self.ap_data_source_name = self.ap_item_name
        if atmospheric_pressure_measurement_name not in list(
            available_ap_measurements.MeasurementName
        ):
            raise ValueError(
                "Atmospheric pressure measurement name "
                f"'{atmospheric_pressure_measurement_name}' not found at"
                f" site '{site}'. "
                "Available measurements are "
                f"{list(available_ap_measurements.MeasurementName)}"
            )

        # Water Temperature
        available_wt_measurements = wt_hilltop.get_measurement_list(site)
        self.water_temperature_measurement_name = water_temperature_measurement_name
        matches = re.search(
            r"([^\[\n]+)(\[(.+)\])?", water_temperature_measurement_name
        )

        if matches is not None:
            self.wt_item_name = matches.groups()[0].strip(" ")
            self.wt_data_source_name = matches.groups()[2]
            if self.wt_data_source_name is None:
                self.wt_data_source_name = self.wt_item_name
        if water_temperature_measurement_name not in list(
            available_wt_measurements.MeasurementName
        ):
            raise ValueError(
                "Water temperature measurement name "
                f"'{water_temperature_measurement_name}' not found at"
                f" site '{site}'. "
                "Available measurements are "
                f"{list(available_wt_measurements.MeasurementName)}"
            )

        self.water_temperature_hts = water_temperature_hts
        self.atmospheric_pressure_hts = atmospheric_pressure_hts

        self.water_temperature_frequency = water_temperature_frequency
        self.atmospheric_pressure_frequency = atmospheric_pressure_frequency

        self.site_altitude = site_altitude
        self.atmospheric_pressure_site_altitude = atmospheric_pressure_site_altitude

        self.ap_standard_item_info = {
            "ItemName": self.ap_item_name,
            "ItemFormat": "F",
            "Divisor": 1,
            "Units": "",
            "Format": "###.##",
        }
        self.wt_standard_item_info = {
            "ItemName": self.wt_item_name,
            "ItemFormat": "F",
            "Divisor": 1,
            "Units": "",
            "Format": "###.##",
        }

        self.ap_standard_data = EMPTY_STANDARD_DATA.copy()
        self.ap_quality_data = EMPTY_QUALITY_DATA.copy()

        self.wt_standard_data = EMPTY_STANDARD_DATA.copy()
        self.wt_quality_data = EMPTY_QUALITY_DATA.copy()

        self.ap_standard_data, _, _, _ = self.import_standard(
            standard_hts=self.atmospheric_pressure_hts,
            site=self.atmospheric_pressure_site,
            standard_measurement_name=self.atmospheric_pressure_measurement_name,
            standard_data_source_name=self.ap_data_source_name,
            standard_item_info=self.standard_item_info,
            standard_data=self.ap_standard_data,
            from_date=self.from_date,
            to_date=self.to_date,
            frequency=self.atmospheric_pressure_frequency,
        )

        self.ap_quality_data, _, _, _ = self.import_quality(
            standard_hts=self.atmospheric_pressure_hts,
            site=self.atmospheric_pressure_site,
            standard_measurement_name=self.atmospheric_pressure_measurement_name,
            standard_data_source_name=self.ap_data_source_name,
            quality_data=self.ap_quality_data,
            from_date=self.from_date,
            to_date=self.to_date,
        )

        self.wt_standard_data, _, _, _ = self.import_standard(
            standard_hts=self.water_temperature_hts,
            site=self.water_temperature_site,
            standard_measurement_name=self.water_temperature_measurement_name,
            standard_data_source_name=self.wt_data_source_name,
            standard_item_info=self.standard_item_info,
            standard_data=self.wt_standard_data,
            from_date=self.from_date,
            to_date=self.to_date,
            frequency=self.water_temperature_frequency,
        )
        self.wt_quality_data, _, _, _ = self.import_quality(
            standard_hts=self.water_temperature_hts,
            site=self.water_temperature_site,
            standard_measurement_name=self.water_temperature_measurement_name,
            standard_data_source_name=self.wt_data_source_name,
            quality_data=self.wt_quality_data,
            from_date=self.from_date,
            to_date=self.to_date,
        )

    @ClassLogger
    def correct_do(
        self, diss_ox=None, atm_pres=None, ap_altitude=None, do_altitude=None
    ):
        """
        Correcting for atmospheric pressure.

        Parameters
        ----------
        diss_ox
        atm_pres
        ap_altitude
        do_altitude

        Returns
        -------
        None, modifies standard_data
        """
        if diss_ox is None:
            diss_ox = self.standard_data
        if atm_pres is None:
            atm_pres = self.ap_standard_data
        if ap_altitude is None:
            ap_altitude = self.atmospheric_pressure_site_altitude
        if do_altitude is None:
            do_altitude = self.site_altitude
        self.standard_data["Value"] = correct_dissolved_oxygen(
            diss_ox["Value"], atm_pres["Value"], ap_altitude, do_altitude
        )

    @ClassLogger
    def quality_encoder(
        self,
        gap_limit: int | None = None,
        max_qc: int | float | None = None,
        interval_dict: dict | None = None,
    ):
        """
        DO verison of quality encoder.

        Parameters
        ----------
        gap_limit
        max_qc
        interval_dict

        Returns
        -------
        None
        """
        super().quality_encoder(
            gap_limit=gap_limit, max_qc=max_qc, interval_dict=interval_dict
        )

        # Atmospheric Pressure
        qc_frame = self.quality_data
        qc_data = compare_two_qc_take_min(
            self.quality_data["Value"], self.ap_quality_data["Value"]
        )

        qc_frame = qc_frame.reindex(qc_data.index, method="ffill")

        diff_idxs = qc_frame[qc_frame["Value"] != qc_data].index

        qc_frame.loc[diff_idxs, "Code"] = "APD"
        qc_frame.loc[diff_idxs, "Details"] = (
            qc_frame.loc[diff_idxs, "Details"] + " [DO QC lowered by AP QC]"
        )
        qc_frame["Value"] = qc_data
        self.quality_data = qc_frame

        # Water temperature
        qc_frame = self.quality_data
        qc_data = compare_two_qc_take_min(
            self.quality_data["Value"], self.wt_quality_data["Value"]
        )

        qc_frame = qc_frame.reindex(qc_data.index, method="ffill")

        diff_idxs = qc_frame[qc_frame["Value"] != qc_data].index

        qc_frame.loc[diff_idxs, "Code"] = "WTD"
        qc_frame.loc[diff_idxs, "Details"] = (
            qc_frame.loc[diff_idxs, "Details"] + " [DO QC lowered by WT QC]"
        )
        qc_frame["Value"] = qc_data
        self.quality_data = qc_frame

        # DO above 100
        cap_frame = cap_qc_where_std_high(
            self.standard_data, self.quality_data, 500, 100
        )
        self._apply_quality(cap_frame)

    @classmethod
    def from_config_yaml(cls, config_path):
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
        processing_parameters = config_yaml_import(config_path)

        ###################################################################################
        # Setting up logging with Annalist
        ###################################################################################

        ann = Annalist()
        ann.configure(
            logfile=processing_parameters["logfile"],
            analyst_name=processing_parameters["analyst_name"],
            stream_format_str=processing_parameters["format"]["stream"],
            file_format_str=processing_parameters["format"]["file"],
        )

        ###################################################################################
        # Creating a Hydrobot Processor object which contains the data to be processed
        ###################################################################################
        now = datetime.now()
        return (
            cls(
                processing_parameters["base_url"],
                processing_parameters["site"],
                processing_parameters["standard_hts_filename"],
                processing_parameters["standard_measurement_name"],
                processing_parameters.get("frequency", None),
                processing_parameters["site_altitude"],
                processing_parameters["water_temperature_site"],
                processing_parameters["atmospheric_pressure_site"],
                processing_parameters["water_temperature_hts"],
                processing_parameters["atmospheric_pressure_hts"],
                processing_parameters["atmospheric_pressure_frequency"],
                processing_parameters["water_temperature_frequency"],
                processing_parameters["atmospheric_pressure_site_altitude"],
                processing_parameters.get("water_temperature_measurement_name", None),
                processing_parameters.get(
                    "atmospheric_pressure_measurement_name", None
                ),
                processing_parameters["from_date"],
                processing_parameters.get("to_date", now.strftime("%d-%m-%Y %H:%M:%S")),
                processing_parameters.get("check_hts_filename", None),
                processing_parameters.get("check_measurement_name", None),
                processing_parameters["defaults"],
                processing_parameters["inspection_expiry"],
                constant_check_shift=processing_parameters["constant_check_shift"],
            ),
            ann,
        )
