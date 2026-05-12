import pytest
import sys
import os
import pandas as pd
import pandas.testing as pdt
from pyspark.sql import SparkSession
from Spark.spark import (
    age_group_analysis,
    average_passengers_crew_cabins,
    high_crew_avg,
    high_pass_avg,
    top_5_CtoP_ratio,
    top_5_tonnage,
    corelation_analysis,
)


@pytest.fixture(scope="session")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    return (
        SparkSession.builder.appName("PySparkUnitTests")
        .master("local[*]")
        .getOrCreate()
    )


@pytest.fixture(scope="session")
def sample_df(spark):
    data = [
        ("Royal", "Ship A", 1000, 600, 500, 5, 50.0),
        ("Royal", "Ship B", 2000, 1000, 1000, 15, 100.0),
        ("Carnival", "Ship C", 3000, 1000, 1500, 25, 150.0),
    ]
    columns = [
        "Cruise_Line",
        "Ship_name",
        "passengers",
        "crew",
        "cabins",
        "Age",
        "Tonnage",
    ]
    return spark.createDataFrame(data, columns)


def test_average_passengers_crew_cabins(sample_df):
    result_df = average_passengers_crew_cabins(sample_df)
    result_pd = (
        result_df.toPandas().sort_values("Cruise_Line").reset_index(drop=True)
    )
    expected_df = pd.DataFrame(
        {
            "Cruise_Line": ["Carnival", "Royal"],
            "average_passengers": [3000.0, 1500.0],
            "average_crew": [1000.0, 800.0],
            "average_cabins": [1500.0, 750.0],
        }
    )
    pdt.assert_frame_equal(result_pd, expected_df, check_exact=False)


def test_high_crew_avg(sample_df):
    result_df = high_crew_avg(sample_df)
    result_pd = result_df.toPandas()
    expected_df = pd.DataFrame(
        {"Cruise_Line": ["Carnival"], "average_crew": [1000.0]}
    )
    pdt.assert_frame_equal(result_pd, expected_df, check_exact=False)


def test_top_5_CtoP_ratio(sample_df):
    result_df = top_5_CtoP_ratio(sample_df)
    result_pd = result_df.toPandas()
    expected_df = pd.DataFrame(
        {
            "Ship_name": ["Ship A", "Ship B", "Ship C"],
            "Cruise_line": ["Royal", "Royal", "Carnival"],
            "crew": [600, 1000, 1000],
            "passengers": [1000, 2000, 3000],
            "CtoP_ratio": [0.6, 0.5, 0.3333],
        }
    )
    pdt.assert_frame_equal(
        result_pd, expected_df, check_exact=False, check_like=True
    )


def test_age_group_analysis(sample_df):
    result_df = age_group_analysis(sample_df)
    result_pd = result_df.toPandas()
    expected_df = pd.DataFrame(
        {
            "Age_Group": ["0-10", "11-20", "21+ Years"],
            "average_passengers": [1000.0, 2000.0, 3000.0],
            "average_crew": [600.0, 1000.0, 1000.0],
        }
    )
    pdt.assert_frame_equal(result_pd, expected_df, check_exact=False)


def test_corelation_analysis(sample_df):
    result_df = corelation_analysis(sample_df)
    result_pd = result_df.toPandas()
    expected_df = pd.DataFrame(
        {
            "Tonnage_Crew_Correlation": [0.87],
            "Passengers_Cabins_Correlation": [1.0],
        }
    )
    pdt.assert_frame_equal(result_pd, expected_df, check_exact=False)


def test_top_5_tonnage(sample_df):
    result_df = top_5_tonnage(sample_df)
    result_pd = result_df.toPandas()
    expected_df = pd.DataFrame(
        {
            "Ship_name": ["Ship C", "Ship B", "Ship A"],
            "Cruise_line": ["Carnival", "Royal", "Royal"],
            "Tonnage": [150.0, 100.0, 50.0],
        }
    )
    pdt.assert_frame_equal(
        result_pd, expected_df, check_exact=False, check_like=True
    )


def test_high_pass_avg(sample_df):
    result_df = high_pass_avg(sample_df)
    result_df = result_df.toPandas()
    expected_df = pd.DataFrame(
        {"Cruise_Line": ["Carnival"], "average_passengers": [3000.0]}
    )
    pdt.assert_frame_equal(result_df, expected_df, check_exact=False)
