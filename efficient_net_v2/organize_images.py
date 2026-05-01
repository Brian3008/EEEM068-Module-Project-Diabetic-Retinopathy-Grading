import os
import csv
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
CSV_PATH = os.path.join(BASE_DIR, "trainLabels.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "src", "data", "processed")
OUTPUT_ROOT = os.path.join(BASE_DIR, "src", "data", "colored_images")

# Map numeric level to folder name
LEVEL_TO_CLASS = {
    "0": "No_DR",
    "1": "Mild",
    "2": "Moderate",
    "3": "Severe",
    "4": "Proliferate_DR",
}

def main():
    # Create class folders
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for cls in LEVEL_TO_CLASS.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

    # Read CSV
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image"]      # e.g., "10_left"
            level = row["level"]         # e.g., "0"

            if level not in LEVEL_TO_CLASS:
                print(f"Skipping {image_id}: unknown level {level}")
                continue

            class_name = LEVEL_TO_CLASS[level]

            # Your processed images are named like "10_left.jpeg"
            src_filename = f"{image_id}.jpeg"
            src_path = os.path.join(PROCESSED_DIR, src_filename)

            if not os.path.isfile(src_path):
                print(f"WARNING: image not found: {src_path}")
                continue

            dst_path = os.path.join(OUTPUT_ROOT, class_name, src_filename)

            shutil.copy2(src_path, dst_path)

    print("Done organizing images.")
    print(f"Organized images in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
