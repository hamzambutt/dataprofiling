import ibis
import os
import pandas as pd
import dagster as dg
from project_ibis.analytic_ibis import (
    avg_pas_crew as apc,
    high_crew_avg as hca,
    high_pass_avg as hpa,
    top_5_CtoP_ratio as t5cr,
    top_5_tonnage as t5t,
    age_group_analysis as aga,
    corelation_analysis as ca,
)


@dg.asset
def raw_cruise_data() -> pd.DataFrame:

    file_path = os.path.join(os.getcwd(), "data", "cruise_dataset.csv")
    conn = ibis.duckdb.connect()
    return conn.read_csv(file_path).execute()


@dg.asset
def avg_pas_crew(raw_cruise_data: pd.DataFrame):

    t = ibis.memtable(raw_cruise_data)
    return apc(t).execute()


@dg.asset
def top_5_CtoP_ratio(raw_cruise_data: pd.DataFrame):

    t = ibis.memtable(raw_cruise_data)
    return t5cr(t).execute()


@dg.asset
def age_group_analysis(raw_cruise_data: pd.DataFrame):

    t = ibis.memtable(raw_cruise_data)
    return aga(t).execute()


@dg.asset
def corelation_analysis(raw_cruise_data: pd.DataFrame):

    t = ibis.memtable(raw_cruise_data)
    return ca(t).execute()


@dg.asset
def high_crew_avg(raw_cruise_data: pd.DataFrame):
    t = ibis.memtable(raw_cruise_data)
    return hca(t).execute()


@dg.asset
def high_pass_avg(raw_cruise_data: pd.DataFrame):
    t = ibis.memtable(raw_cruise_data)
    return hpa(t).execute()


@dg.asset
def top_5_tonnage(raw_cruise_data: pd.DataFrame):
    t = ibis.memtable(raw_cruise_data)
    return t5t(t).execute()
