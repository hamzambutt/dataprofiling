import os
from dynamic import Profiling
from visual import DataVisualizer

if not os.path.exists("data/dataset_2.npy"):
    print("Error: File 'data/dataset_2.npy' not found!")
else:
    profiler = Profiling("data/dataset_2.npy")
    report = profiler.data_stats(p_list=[25, 50, 75])

    # Save the JSON report
    profiler.save_report(report, "reports/stats_summary.json")
    print("Stats report saved.")

    # Visualization
    viz = DataVisualizer(df=profiler.df, stats=report)
    viz.save_plots()
    print("Plots saved to reports/plots/")
