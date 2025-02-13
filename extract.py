import os
import numpy as np

def extract_and_delete_npz(root_folder):
    """
    Recursively finds, extracts, and deletes all 'scene_metadata.npz' files in subdirectories.

    Parameters:
    - root_folder (str): The root directory to search in.
    """
    for dirpath, _, filenames in os.walk(root_folder):
        if "scene_metadata.npz" in filenames:
            npz_path = os.path.join(dirpath, "scene_metadata.npz")
            print(f"Extracting: {npz_path}")

            # Load and save contents
            data = np.load(npz_path)
            for key in data.files:
                save_path = os.path.join(dirpath, f"{key}.npy")
                np.save(save_path, data[key])
                print(f"Saved: {save_path}")

            # Delete the original .npz file
            os.remove(npz_path)
            print(f"Deleted: {npz_path}")

# Example usage:
root_folder = "arkitscenes"  # Replace with your folder path
extract_and_delete_npz(root_folder)
