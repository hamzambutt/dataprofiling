import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, round, when, corr


# Cruise Level Insgihts
def average_passengers_crew_cabins(df):
    return df.groupby("Cruise_Line").agg(
        avg("passengers").alias("average_passengers"),
        avg("crew").alias("average_crew"),
        avg("cabins").alias("average_cabins"),
    )


def high_crew_avg(df):
    return (
        df.groupby("Cruise_Line")
        .agg(avg("crew").alias("average_crew"))
        .orderBy(col("average_crew").desc())
        .limit(1)
    )


def high_pass_avg(df):
    return (
        df.groupby("Cruise_Line")
        .agg(avg("passengers").alias("average_passengers"))
        .orderBy(col("average_passengers").desc())
        .limit(1)
    )


# Ship Level Insights
def top_5_CtoP_ratio(df):
    high_CtoP_ratio = df.withColumn(
        "CtoP_ratio", col("crew") / col("passengers")
    )
    top_5_CtoP = (
        high_CtoP_ratio.select(
            "Ship_name",
            "Cruise_line",
            "crew",
            "passengers",
            round(col("CtoP_ratio"), 4).alias("CtoP_ratio"),
        )
        .orderBy(col("CtoP_ratio").desc())
        .limit(5)
    )
    return top_5_CtoP


def top_5_tonnage(df):

    return (
        df.select("Ship_name", "Cruise_line", "Tonnage")
        .orderBy(col("Tonnage").desc())
        .limit(5)
    )


# Age Analysis
def age_group_analysis(df):
    age_analysis = df.withColumn(
        "Age_Group",
        when(col("Age") <= 10, "0-10")
        .when(col("Age").between(11, 20), "11-20")
        .otherwise("21+ Years"),
    )

    average_age_group = (
        age_analysis.groupby("Age_Group")
        .agg(
            round(avg("passengers"), 2).alias("average_passengers"),
            round(avg("crew"), 2).alias("average_crew"),
        )
        .orderBy(col("Age_Group"))
    )
    return average_age_group


def corelation_analysis(df):
    correlation = df.select(
        round(corr("Tonnage", "crew"), 2).alias("Tonnage_Crew_Correlation"),
        round(corr("passengers", "cabins"), 2).alias(
            "Passengers_Cabins_Correlation"
        ),
    )
    return correlation


if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.appName("TestSetup")
        .master("local[*]")
        .getOrCreate()
    )
    df = spark.read.csv(
        "data/cruise_dataset.csv", header=True, inferSchema=True
    )
    df.show(5)
    print(average_passengers_crew_cabins(df).show())
    print(high_crew_avg(df).show())
    print(high_pass_avg(df).show())
    print(top_5_CtoP_ratio(df).show())
    print(top_5_tonnage(df).show())
    print(age_group_analysis(df).show())
    print(corelation_analysis(df).show())
    spark.stop()
