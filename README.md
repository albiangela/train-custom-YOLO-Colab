# Train Custom YOLO model on Colab

Notebook and helper utilities to fetch datasets, clean/relabel them, tile images, rebalance splits, and train YOLO models end to end.
This is intended to help creating a YOLO-based machine learning model for animal detection and tracking using [`TRex`](https://trex.run/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/albiangela/train-custom-YOLO-Colab/blob/main/Train-custom-YOLO-model-example.ipynb)


## What you can do
- Pull a dataset from Roboflow, a shared Google Drive zip, or the built-in Hexbugs example.
- Collapse/rename annotation labels into a smaller taxonomy and auto-build a compact `data.yaml`.
- Filter out sparse classes, prune empty labels/images, or rebalance the train/valid/test split.
- Tile large images with `yolo-tiler`, optionally keep a fraction of empty tiles, and re-generate labels.
- Summarize class counts, sanity-check splits, zip the prepared dataset, then launch Ultralytics YOLO training, validation, and inference cells.

## Notebook walkthrough
- **Load data**: interactive widget picks Roboflow, Drive zip, or example; downloads/extracts into `/content/datasets`.
- **(Optional) Rename labels**: set a `collapse_map` (`old_id -> new_name`) and `new_class_ids` (`new_name -> new_id`) to merge or drop classes before training.
- **Auto-select classes**: inspect per-class prevalence and suggest `allowed_ids` based on thresholds.
- **Prepare dataset**: run the pipeline to filter labels, prune empty files, tile images (if enabled), and export a clean pool or split set with a new `data.yaml`.
- **Train**: configure Ultralytics YOLO args (model checkpoint, epochs, image size, augmentation) and start training.
- **Evaluate & infer**: pick the best checkpoint by mAP, visualize predictions on validation images, and zip outputs for download.

## Utility modules
All helpers live in `utils/` 

- **Dataset ingestion (`utils/datasets.py`)**
  - `fetch_dataset` accepts `RoboflowSource`, `DriveSource`, or `ExampleSource` and normalizes the folder layout.
  - `launch_dataset_selector` / `DatasetSelector` render ipywidgets to collect credentials/links and set `dataset_path` in the notebook.
  - `prompt_for_dataset` offers a text prompt alternative for CLI/terminal sessions.

- **Preparation & labeling (`utils/prep.py`)**
  - Label tools: `filter_labels`, `simplify_labels`, `build_collapse_map`, `build_new_class_ids_from_yaml`.
  - Dataset pipeline: `prepare_yolo_dataset` (filter -> prune empties -> optional tiling -> optional rebalance) with switches like `do_tile`, `empty_tile_fraction`, `do_rebalance`, and `split`.
  - Tiling: `tile_with_yolo_tiler` wraps `yolo-tiler`, then materializes empty label files and can subsample negatives.
  - Inspection: `summarize_classes`, `auto_select_allowed_ids`, `count_labels`, `check_dataset` for quick distribution checks.
  - YAML: `make_data_yaml` builds a contiguous `names` list for the filtered taxonomy.


Run the notebook cells to follow the same flow with widgets, training commands, and evaluation plots.

This notebook can help with:

	•	Adjust training parameters
	•	Add custom functions
	•	Extend functionality for your specific use case
