# Satellite Image Slicer for YOLO 🛰️

A robust Python toolset designed to process high-resolution (4K) satellite imagery and convert complex JSON annotations into a tiled, training-ready dataset for YOLO object detection models. 

This tool is specifically optimized for OSINT and geographic object recognition tasks, handling the automatic slicing of large images, bounding box recalculation, and smart label parsing.

## ✨ Key Features

* **Smart Image Tiling:** Slices massive 4K satellite images into smaller, model-friendly tiles (e.g., 640x640) with customizable overlap to ensure objects on edges are not lost.
* **Intelligent Label Parsing:** Automatically parses and maps raw, inconsistent labels (e.g., `white_sedan`, `old_muclecar`, `blue_pickup`) into **14 standardized classes** (Heavy Vehicles, Pickups, SUVs, Vans, Hatchbacks, Sedans, Trailers, Buses, Muscle cars, RVs, Junk cars, Construction vehicles, 6x6, Motorcycles).
* **Bounding Box Clipping:** Precisely clips and normalizes coordinates for bounding boxes that intersect tile boundaries.
* **YOLO Format Ready:** Automatically generates the required folder structure (`images/`, `labels/`) and the `data.yaml` configuration file for immediate YOLO training.
* **Dataset Utilities:** Includes additional scripts for verifying bounding boxes visually (`verify_boxes.py`) and managing dataset splits (`merge_and_split.py`).
* **Cyrillic Path Support:** Safe file reading/writing using `numpy` buffers to prevent crashes on Windows machines with non-ASCII file paths.

## 🛠️ Requirements

* `numpy`
* `opencv-python` (cv2)

Install dependencies via pip:
```
pip install numpy opencv-python

🚀 Usage
1. Main Slicer (main_slice.py / Slice.py)

This is the core script. Ensure your input directories (containing raw .jpg/.png images and corresponding .json annotation files) are set correctly in the script configuration.


python main_slice.py

Configuration variables in the script:

    TILE_SIZE: Default is 640.

    OVERLAP: Overlap between tiles in pixels (e.g., 64).

    SAVE_EMPTY_PROB: Probability (0.0 to 1.0) of saving tiles that contain no objects (useful for background training).

2. Verify Bounding Boxes (verify_boxes.py)

After slicing, you can visually inspect if the bounding boxes were mapped and clipped correctly. This script will draw the boxes directly onto the generated tiles.


python verify_boxes.py

3. Merge & Split (merge_and_split.py)

Use this utility to reorganize your dataset or split it into train, val, and test subsets before feeding it into your AI model.


python merge_and_split.py

📂 Project Structure

├── main_slice.py          # Advanced slicer with 14 classes
├── Slice.py               # Alternative/legacy slicer
├── merge_and_split.py     # Dataset train/val splitting logic
├── verify_boxes.py        # Visual validation of YOLO labels
├── test.py                # Sandbox testing scripts
├── .gitignore             # Ignores large datasets and cache
└── README.md              # Project documentation
