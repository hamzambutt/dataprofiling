# flake8: noqa
from PIL import Image, ImageEnhance
import os

folder = "data/current_img"
# Grab all the JPEGs in the folder
images = [
    f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg"))
]

print(f"Injecting Image Drift into {folder}...\n")

if len(images) < 4:
    print("Please put at least 5-10 JPEGs in the current_img folder first!")
else:
    num_to_ruin = max(2, len(images) // 3)

    # 1. VISUAL DRIFT: Make the first batch Black & White
    for filename in images[:num_to_ruin]:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)

            # Capture original metadata
            original_exif = img.info.get("exif")

            converter = ImageEnhance.Color(img)
            bw_img = converter.enhance(0.0)

            # Save while PRESERVING original EXIF
            bw_img.save(img_path, exif=original_exif)
            print(f"Desaturated (Visual Drift): {filename}")
        except Exception as e:
            print(f"Skipping {filename}: Could not open image. ({e})")

    # 2. METADATA DRIFT: Convert the second batch to PNGs
    for filename in images[num_to_ruin : num_to_ruin * 2]:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)

            png_path = os.path.splitext(img_path)[0] + ".png"
            # Metadata will be stripped here during conversion
            img.save(png_path, "PNG")
            os.remove(img_path)
            print(f"Converted to PNG (Metadata Drift): {filename}")
        except Exception as e:
            print(f"Skipping {filename}: Could not open image. ({e})")

    print("\nDrift Injection Complete!")
