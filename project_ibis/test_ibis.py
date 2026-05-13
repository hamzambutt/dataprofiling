import pytest
import ibis
import pandas as pd
import pandas.testing as pdt
from project_ibis.analytic_ibis import (
    avg_pas_crew,
    high_crew_avg,
    high_pass_avg,
    top_5_CtoP_ratio,
    top_5_tonnage,
    age_group_analysis,
    corelation_analysis,
)


@pytest.fixture(scope="session")
def sample_data():
    data = pd.DataFrame(
        {
            "Cruise_line": ["Royal", "Royal", "Carnival"],
            "Ship_name": ["Ship A", "Ship B", "Ship C"],
            "passengers": [1000, 2000, 3000],
            "crew": [500, 1000, 1000],
            "cabins": [500, 1000, 1500],
            "Age": [5, 15, 25],
            "Tonnage": [50.0, 100.0, 150.0],
        }
    )
    return ibis.memtable(data)


def test_average_passengers_crew_cabins(sample_data):
    actual_t = avg_pas_crew(sample_data).execute()

    actual_df = (
        actual_t[["Cruise_line", "average_crew"]]
        .sort_values("Cruise_line")
        .reset_index(drop=True)
    )
    expected_df = pd.DataFrame(
        {"Cruise_line": ["Carnival", "Royal"], "average_crew": [1000.0, 750.0]}
    )
    pdt.assert_frame_equal(actual_df, expected_df, check_exact=True)


def test_high_crew_avg(sample_data):
    actual_t = high_crew_avg(sample_data).execute()

    expecteed_df = pd.DataFrame(
        {"Cruise_line": ["Carnival"], "average_crew": [1000.0]}
    )
    pdt.assert_frame_equal(actual_t, expecteed_df)


def test_top_5_CtoP_ratio(sample_data):
    actual_t = top_5_CtoP_ratio(sample_data).execute()

    expected_df = pd.DataFrame(
        {
            "Ship_name": ["Ship A", "Ship B", "Ship C"],
            "Cruise_line": ["Royal", "Royal", "Carnival"],
            "crew": [500, 1000, 1000],
            "passengers": [1000, 2000, 3000],
            "CtoP_ratio": [0.5, 0.5, 0.3333],
        }
    )
    pdt.assert_frame_equal(actual_t, expected_df)


def test_age_group_analysis(sample_data):
    actual_t = age_group_analysis(sample_data).execute()

    excepted_df = pd.DataFrame(
        {
            "Age_Group": ["0-10", "11-20", "21+ Years"],
            "average_passengers": [1000.0, 2000.0, 3000.0],
            "average_crew": [500.0, 1000.0, 1000.0],
        }
    )
    pdt.assert_frame_equal(actual_t, excepted_df, check_exact=True)


def test_corelation_analysis(sample_data):
    actual_t = corelation_analysis(sample_data).execute()
    expected_df = pd.DataFrame(
        {
            "Tonnage_Crew_Correlation": [0.87],
            "Passengers_Cabins_Correlation": [1.0],
        }
    )
    pdt.assert_frame_equal(actual_t, expected_df, check_exact=True)


def test_high_pass_avg(sample_data):
    actual_t = high_pass_avg(sample_data).execute()
    expected_df = pd.DataFrame(
        {"Cruise_line": ["Carnival"], "average_passengers": [3000.0]}
    )
    pdt.assert_frame_equal(actual_t, expected_df, check_exact=True)


def test_top_5_tonnage(sample_data):
    actual_t = top_5_tonnage(sample_data).execute()
    expected_df = pd.DataFrame(
        {
            "Ship_name": ["Ship C", "Ship B", "Ship A"],
            "Cruise_line": ["Carnival", "Royal", "Royal"],
            "Tonnage": [150.0, 100.0, 50.0],
        }
    )
    pdt.assert_frame_equal(actual_t, expected_df, check_exact=True)
