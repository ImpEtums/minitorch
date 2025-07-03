#!/usr/bin/env python3
"""
Download and extract MNIST data for training.
"""
import urllib.request
import gzip
import os

def download_mnist():
    """Download MNIST dataset files."""
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    for file in files:
        url = base_url + file
        filepath = os.path.join(data_dir, file)
        
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {file}")
        else:
            print(f"{file} already exists")
    
    # Extract gzipped files
    for file in files:
        gz_filepath = os.path.join(data_dir, file)
        filepath = os.path.join(data_dir, file[:-3])  # Remove .gz extension
        
        if not os.path.exists(filepath):
            print(f"Extracting {file}...")
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Extracted {file}")
        else:
            print(f"{file[:-3]} already exists")

if __name__ == "__main__":
    download_mnist()
    print("MNIST data ready!")
