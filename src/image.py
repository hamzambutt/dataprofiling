import os
from metadata import ImageProfiler
from whylogs.viz import NotebookProfileVisualizer


baseline_folder = "data/baseline_img"
production_folder = "data/test"
output_file = "reports/exif_metadata.json"

if not os.path.exists(baseline_folder) or not os.path.exists(
    production_folder
):
    print("Error: 'data/baseline_img' or 'data/test' folder not found.")
    exit()

baseline_prof = ImageProfiler(baseline_folder)
production_prof = ImageProfiler(production_folder)

print("\nGenerating Drift Report...")

visualization = NotebookProfileVisualizer()
visualization.set_profiles(
    target_profile_view=production_prof.profile_view,
    reference_profile_view=baseline_prof.profile_view,
)

os.makedirs("reports", exist_ok=True)
report_path = "reports/image_drift_report.html"

report_html = visualization.summary_drift_report()
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_html.data)

print(f"\nSuccess! Open {report_path} in your web browser.")

production_prof.scan_folder(production_folder, output_file)
