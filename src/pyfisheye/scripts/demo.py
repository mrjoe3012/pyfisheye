import os
import os.path as path
import sys
import zipfile
from tqdm import tqdm
from pyfisheye.scripts.calibrate import calibrate, Arguments
from glob import glob

zipfile_path = 'ImprovedOcamCalib-master.zip'
dataset_img_directory = path.join(
    'ImprovedOcamCalib-master',
    'FisheyeDataSet',
    'test_images'
)
dataset_prefix = [
    'Fisheye1_',
    'Fisheye2_',
    'GOPR'
]
dataset_formats = [
    '.jpg',
    '.jpg',
    '.jpg'
]
dataset_pattern_size = [ # rows, columns, tile width, tile height
    (8, 6, 0.0325, 0.0325),
    (8, 6, 0.117, 0.117),
    (8, 6, 0.117, 0.117)
]
output_directory = 'demo_results'

def print_disclaimer() -> None:
    print("""
    This demonstration script relies on a dataset located in this external repository:
    https://github.com/urbste/ImprovedOcamCalib

    This repository is not affiliated with the pyfisheye project. All rights and copyrights 
    to the dataset remain with the original authors.\n""" + '-' * 96)

def unzip_images() -> None:
    if not os.path.isfile(zipfile_path):
        print(f"Error: Unable to find {zipfile_path}")
        sys.exit(1)
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        # filter to only dataset
        dataset_members = [
            x for x in members if 'FisheyeDataSet' in x.filename
        ]
        for member in tqdm(dataset_members, desc="Extracting", unit="file"):
            zip_ref.extract(member)

def select_dataset() -> int:
    """
    :returns: The selected dataset's index.
    """
    dataset_names = [
        x.replace('_', '') for x in dataset_prefix
    ]
    print("\n".join((f'\t{i}) {x}' for i, x in enumerate(dataset_names))))
    selected_str = input("Please choose a dataset: ")
    try:
        selected_int = int(selected_str)
        if selected_int < 0 or selected_int >= len(dataset_names):
            raise ValueError()
    except ValueError:
        print("Error: Invalid selection.")
        sys.exit(1)
    return selected_int

def run_calibration(index: int) -> None:
    """
    :param index: The index of the dataset to use (see variable at top of script)
    """
    glob_str = path.join(dataset_img_directory, f'{dataset_prefix[index]}*{dataset_formats[index]}')
    images = glob(
        glob_str
    )
    if len(images) == 0:
        print(f"Error: No images found in '{glob_str}'")
        sys.exit(1)
    grid_size_str = input("Choose a grid size (quick=20,accurate=100): ")
    try:
        grid_size = int(grid_size_str)
        if grid_size <= 0:
            raise ValueError()
    except ValueError:
        print("Error: Invalid grid size.")
        sys.exit(1)
    rows, cols, width, height = dataset_pattern_size[index]
    dataset_name = dataset_prefix[index].replace('_', '')
    dataset_output_dir = path.join(output_directory, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    calibration_arguments = Arguments(
        images=images,
        pattern_num_rows=rows,
        pattern_num_cols=cols,
        pattern_tile_width=width,
        pattern_tile_height=height,
        show_corner_det=False,
        save_corner_det=dataset_output_dir,
        optimise_distortion_centre=True,
        initial_distortion_centre_x=None,
        initial_distortion_centre_y=None,
        weighted_least_squares_threshold=1.0,
        num_monotonicity_constraint_samples=1000,
        nonlinear_refinement=True,
        distortion_centre_search_grid_size=grid_size,
        save_results=dataset_output_dir,
        show_results=False,
        save_calibration_to=path.join(dataset_output_dir, 'calibration.json'),
        log_level='INFO'
    )
    calibrate(calibration_arguments)

def demo() -> None:
    unzip_images()
    print_disclaimer()
    index = select_dataset()
    run_calibration(index)

if __name__ == '__main__':
    demo()
