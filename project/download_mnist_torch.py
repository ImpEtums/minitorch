#!/usr/bin/env python3
"""
Download MNIST data using torchvision and convert to the format expected by python-mnist.
"""
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np

def download_and_convert_mnist():
    """Download MNIST and convert to python-mnist format."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download MNIST dataset
    print("Downloading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    
    print("MNIST dataset downloaded successfully!")
    
    # The torchvision MNIST dataset will be in data/MNIST/raw/
    # python-mnist expects files in data/ directory
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    
    # Copy files to the expected location
    files_to_copy = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte", 
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte"
    ]
    
    for file in files_to_copy:
        src = os.path.join(raw_dir, file)
        dst = os.path.join(data_dir, file)
        
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"Copying {file}...")
            with open(src, 'rb') as f_in:
                with open(dst, 'wb') as f_out:
                    f_out.write(f_in.read())
    
    print("MNIST data is ready for use!")

if __name__ == "__main__":
    download_and_convert_mnist()
