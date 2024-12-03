import os
import shutil


def copy_and_flatten_datasets(source_dir, dest_dir, exclude_dirs=["backup"]):
    # Ensure destination subdirectories exist
    for split in ["train", "valid", "test"]:
        for subfolder in ["images", "labels"]:
            os.makedirs(os.path.join(dest_dir, split, subfolder), exist_ok=True)

    # Convert destination directory to absolute path for comparison
    dest_dir_abs = os.path.abspath(dest_dir)

    # Traverse the source directory
    for root, dirs, files in os.walk(source_dir):
        # Exclude destination and other specified directories from traversal
        dirs[:] = [
            d
            for d in dirs
            if os.path.abspath(os.path.join(root, d)) != dest_dir_abs and d not in exclude_dirs
        ]

        # Get the last two parts of the path to identify 'images' or 'labels' folders inside 'train', 'valid', or 'test'
        parts = os.path.normpath(root).split(os.sep)[-2:]
        if len(parts) < 2:
            continue

        split = parts[0]
        subfolder = parts[1]

        # Check if current directory is 'images' or 'labels' inside 'train', 'valid', or 'test'
        if split in ["train", "valid", "test"] and subfolder in ["images", "labels"]:
            # Destination directory (flattened)
            dest_subdir = os.path.join(dest_dir, split, subfolder)

            # Determine dataset name to help with unique naming
            dataset_name = os.path.normpath(root).split(os.sep)[-3]

            # Copy files
            for file_name in files:
                source_file = os.path.join(root, file_name)
                file_base, file_ext = os.path.splitext(file_name)

                # Create a unique file name to avoid collisions
                unique_file_name = f"{dataset_name}_{file_name}"
                dest_file = os.path.join(dest_subdir, unique_file_name)

                # If the file already exists, append a counter
                counter = 1
                while os.path.exists(dest_file):
                    unique_file_name = f"{dataset_name}_{file_base}_{counter}{file_ext}"
                    dest_file = os.path.join(dest_subdir, unique_file_name)
                    counter += 1

                shutil.copy2(source_file, dest_file)
            print(f"Copied from '{root}' to '{dest_subdir}'")

if __name__ == "__main__":
    source_directory = r"D:\SBHNL\Images\BSML\Datasets\Finishing"
    destination_directory = r"D:\SBHNL\Images\BSML\Datasets\Finishing\finishing_united"
    copy_and_flatten_datasets(source_directory, destination_directory)
