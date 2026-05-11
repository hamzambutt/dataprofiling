import ibis
import os


def avg_pas_crew(t):
    return t.group_by("Cruise_line").aggregate(
        average_passengers=t["passengers"].mean(),
        average_crew=t["crew"].mean(),
        average_cabins=t["cabins"].mean(),
    )


def high_crew_avg(t):
    return (
        t.group_by("Cruise_line")
        .aggregate(average_crew=t["crew"].mean())
        .order_by(ibis.desc("average_crew"))
        .limit(1)
    )


def high_pass_avg(t):
    return (
        t.group_by("Cruise_line")
        .aggregate(average_passengers=t["passengers"].mean())
        .order_by(ibis.desc("average_passengers"))
        .limit(1)
    )


def top_5_CtoP_ratio(t):
    tp5 = (
        t.mutate(CtoP_ratio=t["crew"] / t["passengers"])
        .order_by(ibis.desc("CtoP_ratio"))
        .limit(5)
    )

    return (
        tp5.select(
            "Ship_name",
            "Cruise_line",
            "crew",
            "passengers",
            CtoP_ratio=tp5["CtoP_ratio"].round(4),
        )
        .order_by(ibis.desc("CtoP_ratio"))
        .limit(5)
    )


def top_5_tonnage(t):
    return (
        t.select("Ship_name", "Cruise_line", "Tonnage")
        .order_by(ibis.desc("Tonnage"))
        .limit(5)
    )


def age_group_analysis(t):
    age_bucket = ibis.cases(
        (t["Age"] <= 10, "0-10"),
        ((t["Age"] > 10) & (t["Age"] <= 20), "11-20"),
        else_="21+ Years",
    )
    t_mut = t.mutate(Age_Group=age_bucket)
    return (
        t_mut.group_by("Age_Group")
        .aggregate(
            average_passengers=t_mut["passengers"].mean(),
            average_crew=t_mut["crew"].mean(),
        )
        .order_by("Age_Group")
    )


def corelation_analysis(t):
    return t.aggregate(
        Tonnage_Crew_Correlation=t["Tonnage"]
        .corr(t["crew"], how="pop")
        .round(2),
        Passengers_Cabins_Correlation=t["passengers"]
        .corr(t["cabins"], how="pop")
        .round(2),
    )


if __name__ == "__main__":

    file_path = os.path.join(os.getcwd(), "data", "cruise_dataset.csv")
    t = ibis.read_csv(file_path)

    print(avg_pas_crew(t).execute())
    print(high_crew_avg(t).execute())
    print(high_pass_avg(t).execute())
    print(top_5_CtoP_ratio(t).execute())
    print(top_5_tonnage(t).execute())
    print(age_group_analysis(t).execute())
    print(corelation_analysis(t).execute())
