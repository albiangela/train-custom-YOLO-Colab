# Train Custom YOLO on Colab

Notebook and helper utilities to fetch datasets, clean/relabel them, tile images, rebalance splits, and train YOLO models end to end.

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
All helpers live in `utils/` (importable as `train_custom_yolo` if placed on `sys.path`).

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

## Minimal code example
```python
from utils import (
    DriveSource, fetch_dataset,
    build_collapse_map, build_new_class_ids_from_yaml,
    prepare_yolo_dataset, make_data_yaml, check_dataset,
)

# 1) Download data
dataset_path = fetch_dataset(DriveSource(file_id="your_drive_file_id"))

# 2) Collapse labels (optional)
collapse_map = build_collapse_map([0, 1, 3])  # keep raw ids 0,1,3
new_class_ids = {"shark": 0, "sting_ray": 1}

# 3) Prepare + split
prepare_yolo_dataset(
    dataset_path, "prepared_dataset",
    collapse_map={0: "sting_ray", 1: "sting_ray", 3: "shark"},
    new_class_ids=new_class_ids,
    do_tile=True, slice_wh=(640, 640), empty_tile_fraction=0.2,
    do_rebalance=True, split=(0.7, 0.2, 0.1),
)
data_yaml = make_data_yaml("prepared_dataset", new_class_ids)
check_dataset("prepared_dataset")
```

Run the notebook cells to follow the same flow with widgets, training commands, and evaluation plots.
