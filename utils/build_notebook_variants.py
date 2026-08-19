"""Build the maintained Colab notebook variants from the example notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Train-custom-YOLO-model-example.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


COMMON_IMPORTS = """import glob
import os
import random
import shutil
import sys
import time
from pathlib import Path

from google.colab import drive, runtime
from IPython.display import Image, display
"""

PACKAGE_SETUP = """# Run this before importing NumPy, SciPy, Ultralytics, or dataset-fixer.
%pip install --upgrade ultralytics
%pip install "dataset-fixer @ git+https://github.com/mooch443/dataset-fixer.git"

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from dataset_fixer import Dataset as FixedDataset
import numpy as np
"""

UTILS_SETUP = """# The dataset selector loads its helper directly and is safe to run independently.
print('Dataset helper will be loaded by the selector cell.')
"""

DATASET_SELECTOR = """import importlib.util
import subprocess
import sys
from pathlib import Path

helper_candidates = [
    Path.cwd() / 'utils' / 'datasets.py',
    Path('/content/train-custom-YOLO-Colab/utils/datasets.py'),
]
helper_path = next((path.resolve() for path in helper_candidates if path.is_file()), None)

if helper_path is None:
    repository_root = Path('/content/train-custom-YOLO-Colab')
    subprocess.run([
        'git', 'clone',
        'https://github.com/albiangela/train-custom-YOLO-Colab.git',
        str(repository_root),
    ], check=True)
    helper_path = repository_root / 'utils' / 'datasets.py'

spec = importlib.util.spec_from_file_location('_yolo_dataset_selector', helper_path)
datasets_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = datasets_module
spec.loader.exec_module(datasets_module)

workspace_dataset_root = Path('/content/datasets')
print('Dataset selector loaded from:', helper_path)
datasets_module.launch_dataset_selector(globals(), dataset_root=workspace_dataset_root)
"""

RELABEL_DATASET_SELECTOR = """import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

def selector_is_current(module):
    return 'include_multilabel_example' in inspect.signature(module.launch_dataset_selector).parameters

# Load the helper file directly. This avoids requiring `utils` to be an
# importable package and makes this cell independent of earlier setup cells.
candidate_files = [
    Path.cwd() / 'utils' / 'datasets.py',
    Path('/content/train-custom-YOLO-Colab/utils/datasets.py'),
]
candidate_files.extend(
    parent / 'train-custom-YOLO-Colab' / 'utils' / 'datasets.py'
    for parent in Path.cwd().parents
)

available_helpers = [
    path.resolve()
    for path in dict.fromkeys(candidate_files)
    if path.is_file()
]
if not available_helpers:
    repository_root = Path('/content/train-custom-YOLO-Colab')
    subprocess.run([
        'git', 'clone',
        'https://github.com/albiangela/train-custom-YOLO-Colab.git',
        str(repository_root),
    ], check=True)
    available_helpers = [repository_root / 'utils' / 'datasets.py']

datasets_module = None
for helper_path in available_helpers:
    spec = importlib.util.spec_from_file_location('_current_yolo_datasets', helper_path)
    candidate_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate_module
    spec.loader.exec_module(candidate_module)
    if datasets_module is None or selector_is_current(candidate_module):
        datasets_module = candidate_module
    if selector_is_current(candidate_module):
        break

workspace_dataset_root = Path('/content/datasets')
print('Dataset selector loaded from:', datasets_module.__file__)

if selector_is_current(datasets_module):
    datasets_module.launch_dataset_selector(
        globals(),
        dataset_root=workspace_dataset_root,
        include_multilabel_example=True,
    )
else:
    # Backward-compatible unified selector for older Colab checkouts.
    import ipywidgets as widgets
    from IPython.display import display

    style = {'description_width': '120px'}
    source_dropdown = widgets.Dropdown(
        options=[
            ('Roboflow snippet', 'roboflow'),
            ('Google Drive link', 'drive'),
            ('Example dataset (Hexbugs)', 'hexbugs'),
            ('Multiabel example', 'multilabel'),
        ],
        value='roboflow',
        description='Source:',
        style=style,
    )
    roboflow_input = widgets.Textarea(
        description='Snippet:',
        placeholder='Paste the full Roboflow download snippet here…',
        layout=widgets.Layout(width='100%', height='180px'),
        style=style,
    )
    drive_input = widgets.Text(
        description='Drive link:',
        placeholder='https://drive.google.com/file/d/…',
        layout=widgets.Layout(width='100%'),
        style=style,
    )
    drive_name = widgets.Text(
        description='Dataset name:',
        placeholder='Optional folder name',
        layout=widgets.Layout(width='100%'),
        style=style,
    )
    example_help = widgets.HTML()
    download_button = widgets.Button(
        description='Download dataset',
        button_style='primary',
        icon='download',
    )
    selector_status = widgets.Output()

    roboflow_box = widgets.VBox([roboflow_input])
    drive_box = widgets.VBox([drive_input, drive_name])
    forms = {
        'roboflow': roboflow_box,
        'drive': drive_box,
        'hexbugs': example_help,
        'multilabel': example_help,
    }

    def show_selected_form(kind):
        roboflow_box.layout.display = 'flex' if kind == 'roboflow' else 'none'
        drive_box.layout.display = 'flex' if kind == 'drive' else 'none'
        example_help.layout.display = 'block' if kind in {'hexbugs', 'multilabel'} else 'none'
        if kind == 'hexbugs':
            example_help.value = '<small>Downloads the Hexbugs example from TRex-tutorials-data.</small>'
        elif kind == 'multilabel':
            example_help.value = '<small>Downloads YOLO-models/multilabel-example.</small>'

    def source_changed(change):
        show_selected_form(change['new'])
        with selector_status:
            selector_status.clear_output()

    def download_selected_dataset(_button):
        with selector_status:
            selector_status.clear_output()
            download_button.disabled = True
            try:
                choice = source_dropdown.value
                if choice == 'roboflow':
                    snippet = roboflow_input.value.strip()
                    if not snippet:
                        raise ValueError('Paste the Roboflow snippet first.')
                    source = datasets_module.RoboflowSource(
                        **datasets_module._parse_roboflow_snippet(snippet)
                    )
                elif choice == 'drive':
                    file_id = datasets_module._extract_drive_file_id(drive_input.value.strip())
                    if not file_id:
                        raise ValueError('Provide a valid Google Drive link or file ID.')
                    source = datasets_module.DriveSource(
                        file_id=file_id,
                        name_hint=drive_name.value.strip() or None,
                    )
                else:
                    datasets_module._EXAMPLE_SUBDIR = Path(
                        'YOLO-models/multilabel-example'
                        if choice == 'multilabel'
                        else 'YOLO-models/hexbugs-annotation-dataset'
                    )
                    source = datasets_module.ExampleSource()

                dataset_path = datasets_module.fetch_dataset(
                    source,
                    dataset_root=workspace_dataset_root,
                )
                globals()['dataset_path'] = dataset_path
                globals()['name'] = dataset_path.name
                globals()['DATASET_SOURCE'] = source
                globals()['DATASET_ROOT'] = workspace_dataset_root
                print(f'✅ Dataset ready at: {dataset_path}')
                print(f"   Variable `name` set to '{dataset_path.name}'.")
            except Exception as exc:
                print(f'⚠️ {exc}')
            finally:
                download_button.disabled = False

    source_dropdown.observe(source_changed, names='value')
    download_button.on_click(download_selected_dataset)
    show_selected_form(source_dropdown.value)
    display(widgets.VBox([
        source_dropdown,
        roboflow_box,
        drive_box,
        example_help,
        download_button,
        selector_status,
    ]))
"""

COMMON_PREP = """import yaml
from types import SimpleNamespace

dataset = SimpleNamespace(location='/content/datasets/', name=name, version=1)
dataset_root = os.path.join(dataset.location, name)
source_yaml = os.path.join(dataset_root, 'data.yaml')
out_dir = dataset_root + '-dataset-fixed'

assert os.path.isdir(os.path.join(dataset_root, 'train')), dataset_root
assert os.path.isfile(source_yaml), source_yaml

# Build a portable compatibility YAML without modifying the downloaded source.
# This also repairs Roboflow exports that contain paths such as ../train/images.
with open(source_yaml, 'r') as stream:
    fixer_yaml_data = yaml.safe_load(stream) or {}
fixer_yaml_data['path'] = os.path.abspath(dataset_root)
for split_key in ('train', 'val', 'valid', 'validation', 'test'):
    fixer_yaml_data.pop(split_key, None)
for yaml_key, folder_candidates in {
    'train': ('train',),
    'val': ('val', 'valid', 'validation'),
    'test': ('test',),
}.items():
    for folder_name in folder_candidates:
        if os.path.isdir(os.path.join(dataset_root, folder_name, 'images')):
            fixer_yaml_data[yaml_key] = f'{folder_name}/images'
            break

fixer_source_yaml = dataset_root + '-dataset-fixer-input.yaml'
with open(fixer_source_yaml, 'w') as stream:
    yaml.safe_dump(fixer_yaml_data, stream, sort_keys=False, allow_unicode=True)

print('Dataset:', dataset_root)
print('dataset-fixer input:', fixer_source_yaml)
print('Fixed output:', out_dir)
"""

RELABEL_WIDGET = """import ipywidgets as widgets
import yaml

with open(source_yaml, 'r') as stream:
    yaml_data = yaml.safe_load(stream) or {}

raw_names = yaml_data.get('names')
if isinstance(raw_names, dict):
    original_names = [str(raw_names.get(i, raw_names.get(str(i), f'class_{i}')))
                      for i in range(max(map(int, raw_names.keys())) + 1)]
elif isinstance(raw_names, list):
    original_names = [str(name) for name in raw_names]
else:
    raise ValueError(f"No valid 'names' list or dictionary in {source_yaml}")

mapping_rows = []
for class_id, class_name in enumerate(original_names):
    keep = widgets.Checkbox(value=True, description='Keep', indent=False, layout=widgets.Layout(width='75px'))
    target = widgets.Text(value=class_name, description=f'{class_id}: {class_name} →',
                          style={'description_width': 'initial'}, layout=widgets.Layout(width='520px'))
    mapping_rows.append((class_id, keep, target))

status = widgets.HTML('<i>Edit the mapping, then click Apply mapping.</i>')
apply_button = widgets.Button(description='Apply mapping', button_style='success', icon='check')

def apply_label_mapping(_):
    global allowed_ids, collapse_map, new_class_ids
    selected = [(class_id, target.value.strip()) for class_id, keep, target in mapping_rows if keep.value]
    if not selected:
        status.value = '<b style="color:#b00">Keep at least one class.</b>'
        return
    if any(not target_name for _, target_name in selected):
        status.value = '<b style="color:#b00">Target names cannot be empty.</b>'
        return

    ordered_targets = list(dict.fromkeys(target_name for _, target_name in selected))
    new_class_ids = {target_name: new_id for new_id, target_name in enumerate(ordered_targets)}
    collapse_map = {class_id: target_name for class_id, target_name in selected}
    allowed_ids = set(collapse_map)
    status.value = '<b style="color:#080">Mapping ready:</b> ' + ', '.join(
        f'{name}={class_id}' for name, class_id in new_class_ids.items()
    )

apply_button.on_click(apply_label_mapping)
display(widgets.VBox([
    widgets.HTML('<b>Original class → target class</b>'),
    *[widgets.HBox([keep, target]) for _, keep, target in mapping_rows],
    apply_button,
    status,
]))
"""

RELABEL_WITH_FIXER = """def apply_mapping_with_dataset_fixer(fixed_dataset):
    # Temporary unique names make arbitrary rename/merge combinations safe,
    # even when a requested target is also an existing source class name.
    temporary_names = {class_id: f'__source_class_{class_id}__' for class_id in range(len(original_names))}
    plan = fixed_dataset.rename_classes({
        original_names[class_id]: temporary_name
        for class_id, temporary_name in temporary_names.items()
    })

    dropped = [temporary_names[class_id] for class_id in range(len(original_names)) if class_id not in allowed_ids]
    if dropped:
        plan = plan.remove_classes(dropped, visualize=False)

    for target_name in new_class_ids:
        source_ids = [class_id for class_id, mapped_name in collapse_map.items() if mapped_name == target_name]
        source_names = [temporary_names[class_id] for class_id in source_ids]
        anchor, *merged_sources = source_names
        if merged_sources:
            plan = plan.remove_classes(merged_sources, merge_into=anchor, visualize=False)
        plan = plan.rename_classes({anchor: target_name})
    return plan
"""

EXPORT_FIXED_DATASET = """overwrite_output = False
if os.path.exists(out_dir):
    if overwrite_output:
        shutil.rmtree(out_dir)
    else:
        raise FileExistsError(
            f'{out_dir} already exists. Delete it or set overwrite_output=True before rerunning.'
        )

exported = split_plan.export(
    destination=out_dir,
    visualize=True,
    progress=True,
)
exported.assert_trainable()

zip_path = shutil.make_archive(
    out_dir,
    'zip',
    root_dir=os.path.dirname(out_dir),
    base_dir=os.path.basename(out_dir),
)
print('data.yaml:', exported.data_yaml)
print('ZIP written to:', zip_path)
"""

SIMPLE_SECTION = [
    markdown("""# Validate, fix, and split the dataset

This version keeps every original class and uses `dataset-fixer` for validation,
the reproducible train/validation/test split, canonical export, reports, and
`data.yaml` generation. The source dataset is never modified.
"""),
    code(COMMON_PREP),
    code("""fixed_dataset = FixedDataset.open(
    fixer_source_yaml,
    errors='skip',
    progress=True,
)

split_plan = fixed_dataset.split(
    {'train': 0.7, 'val': 0.2, 'test': 0.1},
    seed=42,
    visualize=True,
    progress=True,
)

""" + EXPORT_FIXED_DATASET),
]

RENAME_SECTION = [
    markdown("""# Choose label names

The window reads the original class IDs and names from `data.yaml`. Edit the
target name to rename a class. Give several classes the same target name to
merge them. Uncheck **Keep** to remove a class. Click **Apply mapping** before
running the preparation cell.

`dataset-fixer` validates the source, applies the mapping, creates a
reproducible split, and writes matching labels and `data.yaml` metadata. The
original dataset is never edited.
"""),
    code(COMMON_PREP),
    code(RELABEL_WIDGET),
    code("""
assert 'collapse_map' in globals(), 'Click Apply mapping in the previous cell first.'

fixed_dataset = FixedDataset.open(
    fixer_source_yaml,
    errors='skip',
    progress=True,
)
""" + RELABEL_WITH_FIXER + """
relabelled_plan = apply_mapping_with_dataset_fixer(fixed_dataset)
split_plan = relabelled_plan.split(
    {'train': 0.7, 'val': 0.2, 'test': 0.1},
    seed=42,
    visualize=True,
    progress=True,
)

""" + EXPORT_FIXED_DATASET),
]

TILING_SECTION = [
    markdown("""# Relabel, split, and optionally tile with dataset-fixer

Configure labels exactly as in the relabeling notebook. The dataset is fixed
and split **before** tiling, ensuring tiles derived from one source image remain
in the same cohort. Enable tiling below to create grid or coverage tiles.
"""),
    code(COMMON_PREP),
    code(RELABEL_WIDGET),
    code("""use_tiling = widgets.Checkbox(value=True, description='Create tiles')
tile_mode = widgets.Dropdown(
    options=[('Regular grid', 'grid'), ('Object coverage', 'coverage')],
    value='grid',
    description='Mode:',
)
tile_size = widgets.BoundedIntText(value=640, min=64, max=4096, step=32, description='Tile size:')
tile_overlap = widgets.FloatSlider(value=0.2, min=0.0, max=0.8, step=0.05, description='Overlap:')
background_fraction = widgets.FloatSlider(
    value=0.1, min=0.0, max=0.8, step=0.05, description='Background:'
)

display(widgets.VBox([
    widgets.HTML('<b>Tiling options</b>'),
    use_tiling,
    tile_mode,
    tile_size,
    tile_overlap,
    background_fraction,
]))
"""),
    code("""assert 'collapse_map' in globals(), 'Click Apply mapping in the label cell first.'

fixed_dataset = FixedDataset.open(
    fixer_source_yaml,
    errors='skip',
    progress=True,
)
""" + RELABEL_WITH_FIXER + """
relabelled_plan = apply_mapping_with_dataset_fixer(fixed_dataset)
split_plan = relabelled_plan.split(
    {'train': 0.7, 'val': 0.2, 'test': 0.1},
    seed=42,
    visualize=True,
    progress=True,
)

if use_tiling.value:
    common_tile_options = dict(
        tile_size=tile_size.value,
        seed=42,
        visualize=True,
        errors='skip',
        progress=True,
    )
    if tile_mode.value == 'grid':
        split_plan = split_plan.tile(
            mode='grid',
            overlap=tile_overlap.value,
            negative_tiles=background_fraction.value,
            **common_tile_options,
        )
    else:
        split_plan = split_plan.tile(
            mode='coverage',
            scale_range=(0.75, 1.25),
            target_appearances_per_object=3,
            sparse_appearances_per_object=1,
            background_ratio=background_fraction.value,
            **common_tile_options,
        )

""" + EXPORT_FIXED_DATASET + """

# Compact preview of the actual exported tiles (or images when tiling is off).
exported.visualize(
    split='train',
    samples=6,
    seed=42,
    columns=3,
    panel_size=2.5,
    line_width=2,
)
"""),
]


def build(section: list[dict], title: str, output_name: str) -> None:
    notebook = json.loads(SOURCE.read_text())
    cells = copy.deepcopy(notebook["cells"])
    cells[0] = markdown(title + "\n\n" + "".join(cells[0]["source"]).split("\n", 1)[1])
    cells[2] = code(COMMON_IMPORTS)
    cells[4] = code(PACKAGE_SETUP)
    cells[5] = code(UTILS_SETUP)
    cells[9] = code(DATASET_SELECTOR)
    if output_name in {
        "Train-custom-YOLO-rename-labels.ipynb",
        "Train-custom-YOLO-rename-labels-tiling.ipynb",
    }:
        cells[8] = markdown("""# Load data

Use the selector below to load a dataset from a Roboflow snippet, a shared
Google Drive ZIP, the Hexbugs example, or the **multilabel relabeling example**.

The multilabel example comes from
`TRex-tutorials-data/YOLO-models/multilabel-example` and is intended for trying
the rename, merge, and drop controls in this notebook.
""")
        cells[9] = code(RELABEL_DATASET_SELECTOR)
    for cell in cells[21:]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            'valid_imgs  = os.path.join(out_dir, "valid", "images")',
            'valid_imgs  = os.path.join(out_dir, "val", "images")',
        )
        cell["source"] = source.splitlines(keepends=True)
    notebook["cells"] = cells[:10] + section + cells[21:]
    (ROOT / output_name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build(SIMPLE_SECTION, "# Train a Custom YOLO Model — Simple Dataset Preparation", "Train-custom-YOLO-simple.ipynb")
    build(RENAME_SECTION, "# Train a Custom YOLO Model — Rename or Merge Labels", "Train-custom-YOLO-rename-labels.ipynb")
    build(
        TILING_SECTION,
        "# Train a Custom YOLO Model — Relabel and Tile",
        "Train-custom-YOLO-rename-labels-tiling.ipynb",
    )
