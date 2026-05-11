import pytest
import ibis
import pandas as pd
from Ibis.analytic_ibis import (
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
    result_t = avg_pas_crew(sample_data).execute()

    result = result_t.set_index("Cruise_line")["average_crew"].to_dict()
    assert result["Royal"] == 750.0
    assert result["Carnival"] == 1000.0


def test_high_crew_avg(sample_data):
    result_t = high_crew_avg(sample_data).execute()
    assert len(result_t) == 1
    assert result_t["Cruise_line"][0] == "Carnival"
    assert result_t["average_crew"][0] == 1000.0


def test_top_5_CtoP_ratio(sample_data):
    result_t = top_5_CtoP_ratio(sample_data).execute()
    assert result_t["Ship_name"][0] == "Ship A"
    assert result_t["CtoP_ratio"][0] == 0.5


def test_age_group_analysis(sample_data):
    result_t = age_group_analysis(sample_data).execute()
    result = result_t.set_index("Age_Group")["average_passengers"].to_dict()
    assert result["0-10"] == 1000.0
    assert result["11-20"] == 2000.0
    assert result["21+ Years"] == 3000


def test_corelation_analysis(sample_data):
    result_t = corelation_analysis(sample_data).execute()
    assert "Tonnage_Crew_Correlation" in result_t
    assert "Passengers_Cabins_Correlation" in result_t


def test_high_pass_avg(sample_data):
    result_t = high_pass_avg(sample_data).execute()
    assert len(result_t) == 1
    assert result_t["Cruise_line"][0] == "Carnival"
    assert result_t["average_passengers"][0] == 3000.0


def test_top_5_tonnage(sample_data):
    result_t = top_5_tonnage(sample_data).execute()
    assert result_t["Ship_name"][0] == "Ship C"
    assert result_t["Tonnage"][0] == 150.0
