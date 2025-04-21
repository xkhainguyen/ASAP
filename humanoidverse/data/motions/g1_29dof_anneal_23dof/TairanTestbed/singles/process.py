import joblib
import ipdb
import pathlib
import os

def split_pkl_file(input_file):
    """
    Load a .pkl file and split its keys into separate .pkl files.

    Args:
        input_file (str): Path to the input .pkl file.
        output_files (list): List of output file paths for the split data.
    """
    try:
        # Load the input .pkl file using joblib
        data = joblib.load(input_file)
        print(f"Loaded {len(data)} keys from {input_file}")
        ipdb.set_trace()  # Debugging breakpoint

        # Ensure the number of output files matches the number of keys
        keys = list(data.keys())

        # Split the data and save to separate .pkl files using joblib
        for key in keys:
            joblib.dump({key: data[key]}, f"{key}.pkl")
            print(f"Saved key '{key}' to {key}.pkl")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Input file path
    input_pkl = "khai_leg_move.pkl"

    # Split the .pkl file
    split_pkl_file(input_pkl)
