# pdf_to_images.py
# Convert PDF pages to images for the housing dataset

import os
from pdf2image import convert_from_path

# PDF file path
pdf_path = "houses.pdf"

# Output folder for images
output_folder = "data/house_images"
os.makedirs(output_folder, exist_ok=True)

# Poppler path (Windows)
poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"

# Convert PDF to images
pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)

for i, page in enumerate(pages):
    # Save each page as a JPEG image
    image_filename = os.path.join(output_folder, f"{i+1}.jpg")
    page.save(image_filename, "JPEG")
    print(f"Saved {image_filename}")

print(f"Conversion complete. Total pages: {len(pages)}")
