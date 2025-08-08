import zipfile
from pathlib import Path

# Path to your zip file
zip_path = Path("EcoSpold01.zip")

# Where to extract
output_dir = Path("ecospold_xml")
output_dir.mkdir(parents=True, exist_ok=True)

# Unpack
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Extracted XML files to {output_dir}")
