import pytest
import sys
import os
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
    # Force Spark to use the same Python executable as your current terminal
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
        ("Royal", "Ship A", 1000, 500, 500, 5, 50.0),
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
    result = {
        row["Cruise_Line"]: row["average_crew"] for row in result_df.collect()
    }

    # Royal has 500 and 1000 crew, average should be 750
    assert result["Royal"] == 750.0
    assert result["Carnival"] == 1000.0


def test_high_crew_avg(sample_df):
    result_df = high_crew_avg(sample_df)
    rows = result_df.collect()

    assert len(rows) == 1
    assert rows[0]["Cruise_Line"] == "Carnival"
    assert rows[0]["average_crew"] == 1000.0


def test_top_5_CtoP_ratio(sample_df):
    result_df = top_5_CtoP_ratio(sample_df)
    rows = result_df.collect()

    # Ship A has 500 crew and 1000 passengers (0.5 ratio)
    assert rows[0]["Ship_name"] == "Ship A"
    assert rows[0]["CtoP_ratio"] == 0.5


def test_age_group_analysis(sample_df):
    result_df = age_group_analysis(sample_df)
    result = {
        row["Age_Group"]: row["average_passengers"]
        for row in result_df.collect()
    }

    # Ship A is 5 years old (0-10 bucket) with 1000 passengers
    assert result["0-10"] == 1000.0
    # Ship B is 15 years old (11-20 bucket) with 2000 passengers
    assert result["11-20"] == 2000.0
    # Ship C is 25 years old (21+ bucket) with 3000 passengers
    assert result["21+ Years"] == 3000.0


def test_corelation_analysis(sample_df):
    result_df = corelation_analysis(sample_df)
    row = result_df.collect()[0]

    # With only 3 data points, correlation may not be meaningful,
    # but we can check if it runs without error
    assert "Tonnage_Crew_Correlation" in row.asDict()
    assert "Passengers_Cabins_Correlation" in row.asDict()


def test_top_5_tonnage(sample_df):
    result_df = top_5_tonnage(sample_df)
    rows = result_df.collect()

    # Ship C has the highest tonnage of 150.0
    assert rows[0]["Ship_name"] == "Ship C"
    assert rows[0]["Tonnage"] == 150.0


def test_high_pass_avg(sample_df):
    result_df = high_pass_avg(sample_df)
    rows = result_df.collect()

    assert len(rows) == 1
    assert rows[0]["Cruise_Line"] == "Carnival"
    assert rows[0]["average_passengers"] == 3000
