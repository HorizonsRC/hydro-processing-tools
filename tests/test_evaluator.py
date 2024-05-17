"""Test the evaluator module."""

# import warnings
# pyright: reportGeneralTypeIssues=false

import numpy as np
import pandas as pd
import pytest
from annalist.annalist import Annalist

import hydrobot.data_sources as data_sources
import hydrobot.evaluator as evaluator

ann = Annalist()
ann.configure()

raw_data_dict = {
    pd.Timestamp("2021-01-01 00:00"): 1.0,
    pd.Timestamp("2021-01-01 00:15"): 2.0,
    pd.Timestamp("2021-01-01 00:30"): 10.0,
    pd.Timestamp("2021-01-01 00:45"): 4.0,
    pd.Timestamp("2021-01-01 01:00"): 5.0,
}

gap_data_dict = {
    pd.Timestamp("2021-01-01 00:00"): np.NaN,
    pd.Timestamp("2021-01-01 00:15"): 2.0,
    pd.Timestamp("2021-01-01 00:30"): np.NaN,
    pd.Timestamp("2021-01-01 00:45"): np.NaN,
    pd.Timestamp("2021-01-01 01:00"): 5.0,
    pd.Timestamp("2021-01-01 01:15"): np.NaN,
    pd.Timestamp("2021-01-01 01:30"): np.NaN,
    pd.Timestamp("2021-01-01 01:45"): np.NaN,
    pd.Timestamp("2021-01-01 02:00"): 0.0,
    pd.Timestamp("2021-01-01 02:15"): 0.0,
    pd.Timestamp("2021-01-01 02:30"): np.NaN,
    pd.Timestamp("2021-01-01 02:45"): np.NaN,
}

check_data_dict = {
    pd.Timestamp("2021-01-01 00:15"): 2.1,
    pd.Timestamp("2021-01-01 00:45"): 7.0,
    pd.Timestamp("2021-01-01 01:00"): 25.0,
}

qc_data_dict = {
    pd.Timestamp("2021-01-01 00:00"): 600,
    pd.Timestamp("2021-01-01 00:15"): 500,
    pd.Timestamp("2021-01-01 00:45"): 600,
    pd.Timestamp("2021-01-01 01:00"): 500,
    pd.Timestamp("2021-01-01 01:30"): 400,
    pd.Timestamp("2021-01-01 02:15"): 600,
    pd.Timestamp("2021-01-01 02:45"): 600,
}


@pytest.fixture()
def raw_data():
    """Example data for testing. Do not change these values."""
    data_series = pd.Series(raw_data_dict)
    raw_frame = pd.DataFrame({})
    raw_frame["Value"] = data_series
    return raw_frame


@pytest.fixture()
def gap_data():
    """Example data for testing. Do not change these values."""
    data_series = pd.Series(gap_data_dict)
    gap_frame = pd.DataFrame({})
    gap_frame["Value"] = data_series
    return gap_frame


@pytest.fixture()
def check_data():
    """Example data for testing. Do not change these values."""
    data_series = pd.Series(check_data_dict)
    check_frame = pd.DataFrame({})
    check_frame["Value"] = data_series
    return check_frame


@pytest.fixture()
def qc_data():
    """Example data for testing. Do not change these values."""
    data_series = pd.Series(qc_data_dict)
    qc_frame = pd.DataFrame({})
    qc_frame["Value"] = data_series
    qc_frame["Code"] = "CHK"
    qc_frame["Details"] = "Message goes here"
    return qc_frame


@pytest.mark.dependency(name="test_gap_finder")
def test_gap_finder(raw_data, gap_data):
    """Test the gap finder function."""
    no_gap_list = evaluator.gap_finder(raw_data["Value"])
    assert no_gap_list == [], "Gap found where there should be no gap"

    gap_list = evaluator.gap_finder(gap_data["Value"])
    assert len(gap_list) >= 4, "gap_finder did not find one or more of the gaps"
    assert len(gap_list) <= 4, "gap_finder found too many gaps"
    assert gap_list[0][1] == 1, "gap length of 1 not calculated correctly"
    assert gap_list[1][1] == 2, "gap length of 2 not calculated correctly"
    assert gap_list[2][1] == 3, "gap length of 3 not calculated correctly"


@pytest.mark.dependency(name="test_small_gap_closer")
def test_small_gap_closer(raw_data, gap_data):
    """Test the small gap closer function."""
    # No gaps here, nothing should happen
    no_gaps = evaluator.small_gap_closer(raw_data["Value"], 1)
    assert no_gaps.equals(
        raw_data["Value"]
    ), "Data without gaps should not be modified, but was"

    # All gaps should be closed
    removed_gaps = evaluator.small_gap_closer(gap_data["Value"], 5)
    assert len(removed_gaps) == 4, "Data changed during gap closing"

    # Should still have one gap of len 3, others closed
    some_gaps = evaluator.small_gap_closer(gap_data["Value"], 2)
    assert len(some_gaps) == 7, "gap_finder did not find one or more of the gaps"


@pytest.mark.dependency(depends=["test_gap_finder", "test_small_gap_closer"])
def test_small_gap_closer_part2(gap_data):
    """Test the small gap closer some more."""
    # All gaps should be closed
    removed_gaps = evaluator.small_gap_closer(gap_data["Value"], 5)
    assert evaluator.gap_finder(removed_gaps) == [], "Gap not closed!"

    # Should still have one gap of len 3, others closed
    some_gaps = evaluator.small_gap_closer(gap_data["Value"], 2)
    assert len(evaluator.gap_finder(some_gaps)) < 2, "Not enough gaps were removed"
    assert len(evaluator.gap_finder(some_gaps)) > 0, "Too many gaps were removed"
    assert (
        evaluator.gap_finder(some_gaps)[0][1] == 3
    ), "incorrect gap length after gap closure"


def test_find_nearest_time(gap_data):
    """Test the find nearest time function."""
    assert evaluator.find_nearest_time(
        gap_data, pd.Timestamp("2021-01-01 01:00")
    ) == pd.Timestamp("2021-01-01 01:00"), "does not find exact value"
    assert evaluator.find_nearest_time(
        gap_data, pd.Timestamp("2021-01-01 01:07")
    ) == pd.Timestamp("2021-01-01 01:00"), "does not round down correctly"
    assert evaluator.find_nearest_time(
        gap_data, pd.Timestamp("2021-01-01 01:08")
    ) == pd.Timestamp(
        "2021-01-01 01:15"
    ), "does not round up or does not round up to null correctly"
    assert evaluator.find_nearest_time(
        gap_data, pd.Timestamp("2021-01-01 01:23")
    ) == pd.Timestamp("2021-01-01 01:30"), "middle of nulls not reading correctly"
    assert evaluator.find_nearest_time(
        gap_data, pd.Timestamp("2021-01-01 00:00")
    ) == pd.Timestamp("2021-01-01 00:00"), "start time not accepted"
    assert evaluator.find_nearest_time(
        gap_data, pd.Timestamp("2021-01-01 02:45")
    ) == pd.Timestamp("2021-01-01 02:45"), "end time not accepted"
    with pytest.raises(KeyError):
        # out of range forwards
        evaluator.find_nearest_time(gap_data, pd.Timestamp("2021-01-01 02:46"))
    with pytest.raises(KeyError):
        # out of range backwards
        evaluator.find_nearest_time(gap_data, pd.Timestamp("2020-12-31 23:59"))


def test_find_nearest_valid_time(gap_data, qc_data):
    """Test the finc nearest valid time function."""
    assert evaluator.find_nearest_valid_time(
        qc_data, pd.Timestamp("2021-01-01 01:00")
    ) == pd.Timestamp("2021-01-01 01:00"), "does not find exact value"
    assert evaluator.find_nearest_valid_time(
        qc_data, pd.Timestamp("2021-01-01 01:07")
    ) == pd.Timestamp("2021-01-01 01:00"), "does not round down correctly"
    assert evaluator.find_nearest_valid_time(
        qc_data, pd.Timestamp("2021-01-01 01:08")
    ) == pd.Timestamp(
        "2021-01-01 01:00"
    ), "does not round down when round up would lead to null"
    assert evaluator.find_nearest_valid_time(
        gap_data, pd.Timestamp("2021-01-01 01:23")
    ) == pd.Timestamp("2021-01-01 01:00"), "middle of nulls not reading correctly"
    assert evaluator.find_nearest_valid_time(
        qc_data, pd.Timestamp("2021-01-01 00:00")
    ) == pd.Timestamp("2021-01-01 00:00"), "start time not accepted"
    assert evaluator.find_nearest_valid_time(
        qc_data, pd.Timestamp("2021-01-01 02:45")
    ) == pd.Timestamp("2021-01-01 02:45"), "end time not accepted"
    with pytest.raises(KeyError):
        # out of range forwards
        evaluator.find_nearest_valid_time(gap_data, pd.Timestamp("2021-01-01 02:46"))
    with pytest.raises(KeyError):
        # out of range backwards
        evaluator.find_nearest_valid_time(gap_data, pd.Timestamp("2020-12-31 23:59"))


def test_check_data_quality_code(raw_data, check_data):
    """Test check data quality code function."""
    meas = data_sources.QualityCodeEvaluator(10, 2.0)
    with pytest.warns(Warning):
        assert (
            len(
                evaluator.check_data_quality_code(
                    raw_data["Value"], pd.Series({}), meas
                )
            )
            == 1
        ), "fails for empty check data series"
    output = evaluator.check_data_quality_code(
        raw_data["Value"], check_data["Value"], meas
    )
    assert (
        len(output) == len(check_data) + 1
    ), "different amount of QC values than check data values"
    assert output.iloc[0, 0] == 600, "QC 600 test failed"
    assert output.iloc[1, 0] == 500, "QC 500 test failed"
    assert output.iloc[2, 0] == 400, "QC 400 test failed"
    assert output.iloc[3, 0] == 0, "Should have QC 0 at the end"


def test_base_data_qc_filter(gap_data, qc_data):
    """Test base data QC filter function."""
    assert len(
        evaluator.base_data_qc_filter(
            gap_data["Value"],
            pd.Series({pd.Timestamp("2021-01-01 00:00"): True}),
        )
    ) == len(gap_data), "True filter removed data"
    assert (
        len(
            evaluator.base_data_qc_filter(
                gap_data["Value"],
                pd.Series({pd.Timestamp("2021-01-01 00:00"): False}),
            )
        )
        == 0
    ), "False filter did not delete all data"
    data_600 = evaluator.base_data_qc_filter(gap_data["Value"], qc_data["Value"] == 600)
    assert pd.Timestamp("2021-01-01 00:00") in data_600
    assert pd.Timestamp("2021-01-01 00:15") not in data_600
    assert pd.Timestamp("2021-01-01 00:30") not in data_600
    assert pd.Timestamp("2021-01-01 00:45") in data_600
    assert pd.Timestamp("2021-01-01 01:00") not in data_600
    assert pd.Timestamp("2021-01-01 01:15") not in data_600
    assert pd.Timestamp("2021-01-01 02:30") in data_600
    assert pd.Timestamp("2021-01-01 02:45") in data_600

    data_500 = evaluator.base_data_qc_filter(gap_data["Value"], qc_data["Value"] == 500)
    assert pd.Timestamp("2021-01-01 00:00") not in data_500
    assert pd.Timestamp("2021-01-01 00:15") in data_500
    assert pd.Timestamp("2021-01-01 00:30") in data_500
    assert pd.Timestamp("2021-01-01 00:45") not in data_500
    assert pd.Timestamp("2021-01-01 01:00") in data_500
    assert pd.Timestamp("2021-01-01 01:15") in data_500
    assert pd.Timestamp("2021-01-01 02:30") not in data_500
    assert pd.Timestamp("2021-01-01 02:45") not in data_500


def test_base_data_meets_qc(gap_data, qc_data):
    """Test the base data meets QC function."""
    assert len(
        evaluator.base_data_meets_qc(
            gap_data["Value"],
            pd.Series({pd.Timestamp("2021-01-01 00:00"): 600}),
            600,
        )
    ) == len(gap_data), "True filter removed data"
    assert (
        len(
            evaluator.base_data_meets_qc(
                gap_data["Value"],
                pd.Series({pd.Timestamp("2021-01-01 00:00"): 500}),
                600,
            )
        )
        == 0
    ), "False filter did not delete all data"
    data_600 = evaluator.base_data_meets_qc(gap_data["Value"], qc_data["Value"], 600)
    assert pd.Timestamp("2021-01-01 00:00") in data_600
    assert pd.Timestamp("2021-01-01 00:15") not in data_600
    assert pd.Timestamp("2021-01-01 00:30") not in data_600
    assert pd.Timestamp("2021-01-01 00:45") in data_600
    assert pd.Timestamp("2021-01-01 01:00") not in data_600
    assert pd.Timestamp("2021-01-01 01:15") not in data_600
    assert pd.Timestamp("2021-01-01 02:30") in data_600
    assert pd.Timestamp("2021-01-01 02:45") in data_600

    data_500 = evaluator.base_data_meets_qc(gap_data["Value"], qc_data["Value"], 500)
    assert pd.Timestamp("2021-01-01 00:00") not in data_500
    assert pd.Timestamp("2021-01-01 00:15") in data_500
    assert pd.Timestamp("2021-01-01 00:30") in data_500
    assert pd.Timestamp("2021-01-01 00:45") not in data_500
    assert pd.Timestamp("2021-01-01 01:00") in data_500
    assert pd.Timestamp("2021-01-01 01:15") in data_500
    assert pd.Timestamp("2021-01-01 02:30") not in data_500
    assert pd.Timestamp("2021-01-01 02:45") not in data_500


def test_missing_data_quality_code(gap_data, qc_data):
    """Test the missing data quality code function."""
    no_gap_qc = evaluator.missing_data_quality_code(
        gap_data["Value"].fillna(3), qc_data, gap_limit=0
    )
    assert no_gap_qc.equals(qc_data), "QC data modified when there are no gaps"

    new_qc = evaluator.missing_data_quality_code(
        gap_data["Value"], qc_data, gap_limit=0
    )
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 00:00"), "Value"] == 100
    ), "Starting gap not added"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 00:00"), "Code"] == "GAP"
    ), "Starting gap code not correctly added"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 00:00"), "Details"]
        == "Missing data amounting to 0 days 00:15:00"
    ), "Starting gap details not correctly added"

    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 00:15"), "Value"] == 500
    ), "Starting gap not closed"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 00:15"), "Code"] == "CHK"
    ), "Starting gap previous code not correctly continued"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 00:15"), "Details"]
        == "Message goes here [End of gap which started at 2021-01-01 00:00:00]"
    ), "End of gap details not correctly added"

    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 01:15"), "Value"] == 100
    ), "Mid-gap not added"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 01:15"), "Code"] == "GAP"
    ), "Mid gap code not added correctly"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 01:15"), "Details"]
        == "Missing data amounting to 0 days 00:45:00"
    ), "Mid gap code not added correctly"

    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 02:00"), "Value"] == 400
    ), "Gap in middle not closed"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 02:00"), "Code"] == "CHK"
    ), "Mid gap previous code not added correctly"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 02:00"), "Details"]
        == "End of gap which started at 2021-01-01 01:15:00. Returning to QC code first assigned at 2021-01-01 01:30:00"
    ), "Mid gap closing details not correct"

    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 02:30"), "Value"] == 100
    ), "Ending gap not added"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 02:30"), "Code"] == "GAP"
    ), "Ending gap code not added correctly"
    assert (
        new_qc.loc[pd.Timestamp("2021-01-01 02:30"), "Details"]
        == "Missing data amounting to 0 days 00:15:00"
    ), "End of gap details not correctly added"


def test_max_qc_limiter(qc_data):
    """Test max_QC_limiter."""
    limited_qcs = evaluator.max_qc_limiter(qc_data, 500)
    assert (limited_qcs["Value"] <= 500).all(), "maximum not enforced"
    assert len(limited_qcs) == len(qc_data), "data going missing or being added"
    assert (limited_qcs.index == qc_data.index).all(), "index modified"

    # Should downgrade
    assert limited_qcs.loc["2021-01-01 00:00:00", "Code"] == "CHK, LIM"
    assert (
        limited_qcs.loc["2021-01-01 00:00:00", "Details"]
        == "Message goes here [Site QC limit applies to a maximum of 500.]"
    )

    # Should not downgrade
    assert limited_qcs.loc["2021-01-01 00:15:00", "Code"] == "CHK"
    assert limited_qcs.loc["2021-01-01 00:15:00", "Details"] == "Message goes here"

    unlimited_power = evaluator.max_qc_limiter(qc_data, 600)
    assert (unlimited_power["Value"] == qc_data["Value"]).all(), "maximum over enforced"
    assert len(unlimited_power) == len(qc_data), "data going missing or being added"
    assert (unlimited_power.index == qc_data.index).all(), "index modified"

    whoops_maybe_the_power_had_a_limit = evaluator.max_qc_limiter(
        pd.DataFrame({}, columns=["Value", "Code", "Details"]), 600
    )
    assert len(whoops_maybe_the_power_had_a_limit) == 0, "trivial case"


def test_single_downgrade_out_of_validation(gap_data):
    """Test downgrade_out_of_validation."""
    print(gap_data)
    assert gap_data.equals(
        evaluator.single_downgrade_out_of_validation(
            gap_data, gap_data["Value"], pd.DateOffset(minutes=20), 200, False
        )
    ), "changes made when data is good and should be untouched"

    data_with_holes = gap_data.dropna() * 1000
    downgraded_data = evaluator.single_downgrade_out_of_validation(
        data_with_holes,
        data_with_holes["Value"],
        pd.DateOffset(minutes=20),
        200,
        False,
    )

    assert downgraded_data.loc["2021-01-01 00:35", "Value"] == 200
    assert downgraded_data.loc["2021-01-01 00:35", "Code"] == "OOV"
    assert (
        downgraded_data.loc["2021-01-01 00:35", "Details"]
        == "Site inspection overdue. Last inspection at 2021-01-01 00:15:00. Data downgraded to QC200 until next inspection."
    )

    assert downgraded_data.loc["2021-01-01 01:00", "Value"] == 5000
    assert pd.isna(downgraded_data.loc["2021-01-01 01:00", "Code"])
    assert pd.isna(downgraded_data.loc["2021-01-01 01:00", "Details"])

    assert downgraded_data.loc["2021-01-01 01:20", "Value"] == 200
    assert downgraded_data.loc["2021-01-01 01:20", "Code"] == "OOV"
    assert (
        downgraded_data.loc["2021-01-01 01:20", "Details"]
        == "Site inspection overdue. Last inspection at 2021-01-01 01:00:00. Data downgraded to QC200 until next inspection."
    )


def test_cap_qc_where_std_high():
    """Test cap_qc_where_std_high."""
    std = pd.DataFrame(
        {
            "Value": pd.Series(
                {
                    "2021-01-01 01:00": 99,
                    "2021-01-01 02:00": 101,
                    "2021-01-01 03:00": 103,
                    "2021-01-01 04:00": 101,
                    "2021-01-01 05:00": 99,
                    "2021-01-01 06:00": 98,
                    "2021-01-01 07:00": 101,
                    "2021-01-01 08:00": 99,
                    "2021-01-01 09:00": 97,
                }
            ),
            "Code": {
                "2021-01-01 01:00": "ABC",
                "2021-01-01 02:00": "ABC",
                "2021-01-01 03:00": "ABC",
                "2021-01-01 04:00": "ABC",
                "2021-01-01 05:00": "ABC",
                "2021-01-01 06:00": "ABC",
                "2021-01-01 07:00": "ABC",
                "2021-01-01 08:00": "ABC",
                "2021-01-01 09:00": "ABC",
            },
            "Details": {
                "2021-01-01 01:00": "ABC",
                "2021-01-01 02:00": "ABC",
                "2021-01-01 03:00": "ABC",
                "2021-01-01 04:00": "ABC",
                "2021-01-01 05:00": "ABC",
                "2021-01-01 06:00": "ABC",
                "2021-01-01 07:00": "ABC",
                "2021-01-01 08:00": "ABC",
                "2021-01-01 09:00": "ABC",
            },
        }
    )
    qc = pd.DataFrame(
        {
            "Value": pd.Series(
                {
                    "2021-01-01 00:00": 600,
                }
            ),
            "Code": {
                "2021-01-01 00:00": "ABC",
            },
            "Details": {
                "2021-01-01 00:00": "ABC",
            },
        }
    )

    fin = pd.Series(
        {
            "2021-01-01 00:00": 600,
            "2021-01-01 02:00": 500,
            "2021-01-01 05:00": 600,
            "2021-01-01 07:00": 500,
            "2021-01-01 08:00": 600,
        }
    )

    assert fin.equals(evaluator.cap_qc_where_std_high(std, qc, 500, 100)["Value"])
