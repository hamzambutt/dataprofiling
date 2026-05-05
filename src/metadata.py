import json
import os
from PIL import Image
from PIL.ExifTags import TAGS
from whylogs.extras.image_metric import log_image


class ImageProfiler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.profile_view = self.profile_folder()

    def profile_folder(self):
        merge_profile_view = None

        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                image_path = os.path.join(self.folder_path, filename)
                try:
                    img = Image.open(image_path)

                    result = log_image(img)
                    profile = result.profile()

                    exif_data = (
                        img._getexif() if hasattr(img, "_getexif") else None
                    )
                    if exif_data:
                        meta_dict = {
                            TAGS.get(k, k): str(v)
                            for k, v in exif_data.items()
                            if k in TAGS and isinstance(v, (str, int, float))
                        }

                        profile.track(row=meta_dict)
                        print(
                            f"Processed {filename} (Found"
                            "{len(meta_dict)} EXIF tags)"
                        )
                    else:
                        print(f"Processed {filename} (No EXIF metadata found)")

                    image_view = profile.view()

                    if merge_profile_view is None:
                        merge_profile_view = image_view
                    else:
                        merge_profile_view = merge_profile_view.merge(
                            image_view
                        )

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return merge_profile_view

    def get_exif_data(self, image_path):
        try:
            img = Image.open(image_path)

            if not hasattr(img, "_getexif"):
                return None

            exif_raw = img._getexif()
            if not exif_raw:
                return None

            readable_metadata = {}
            for tag_id, value in exif_raw.items():
                tag_name = TAGS.get(tag_id, tag_id)

                if isinstance(value, (str, int, float)):
                    readable_metadata[tag_name] = value

            return readable_metadata

        except Exception as e:
            print(f"Error opening {image_path}: {e}")
            return None

    def scan_folder(self, folder_path, output_path):
        metadata_summary = {}

        if not os.path.exists(folder_path):
            print("Folder not found.")
            return

        folder_data = {}
        images_found = False
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                images_found = True
                filepath = os.path.join(folder_path, filename)

                print(f"\nFile: {filename}")
                metadata = self.get_exif_data(filepath)

                if metadata:
                    folder_data[filename] = metadata
                    print(f"Found {len(metadata)} EXIF tags.")
                else:
                    folder_data[filename] = "No EXIF metadata found."

        if not images_found:
            print("No images found in this folder.")
        metadata_summary[folder_path] = folder_data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_summary, f, indent=4)
