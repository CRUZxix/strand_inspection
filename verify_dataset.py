
from pathlib import Path
import zipfile
import pandas as pd
import cv2

BASE = Path(__file__).resolve().parent
IMAGE_DIR = BASE / "images"
LABELS_FILE = BASE / "labels.csv"
PACKETS_FILE = BASE / "packets.csv"
OUTPUT_DIR = BASE / "verification"
OUTPUT_DIR.mkdir(exist_ok=True)

labels = pd.read_csv(LABELS_FILE)
packets = pd.read_csv(PACKETS_FILE)

print("Images folder:", IMAGE_DIR)
print("Label rows:", len(labels))
print("Packet rows:", len(packets))

# Find image files, including inside subfolders.
image_files = list(IMAGE_DIR.rglob("*"))
image_files = [
    p for p in image_files
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
]
image_lookup = {p.name: p for p in image_files}

print("Image files found:", len(image_files))

# Select up to 12 images for an initial visual check.
selected = labels["image_filename"].drop_duplicates().head(12)

for filename in selected:
    path = image_lookup.get(filename)

    if path is None:
        print("MISSING:", filename)
        continue

    image = cv2.imread(str(path))
    if image is None:
        print("UNREADABLE:", filename)
        continue

    # Mark component positions using their row and column labels.
    records = labels[labels["image_filename"] == filename]

    for _, record in records.iterrows():
        row = int(record["row"])
        col = int(record["column"])

        # These markers indicate expected positions only;
        # they are NOT detected component boundaries.
        text = f"R{row} C{col}"
        cv2.putText(
            image, text, (40 + (col - 1) * 250,
                          60 + (row - 1) * 250),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )

    # Resize for convenient viewing.
    height, width = image.shape[:2]
    scale = min(1200 / width, 900 / height)
    preview = cv2.resize(
        image, (int(width * scale), int(height * scale))
    )

    output_name = Path(filename).stem + "_preview.jpg"
    cv2.imwrite(str(OUTPUT_DIR / output_name), preview)

print("Previews saved to:", OUTPUT_DIR)
print("Open the previews and check whether the markers align.")
