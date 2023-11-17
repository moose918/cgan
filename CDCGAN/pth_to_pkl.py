import os
import glob
import torch
import pickle

# Path to the directory containing .pth files
directory_path = os.getcwd()

# Iterate through all .pth files in the directory
for filename in glob.glob(os.path.join(directory_path, '*.pth')):
    # Load the PyTorch model from .pth file
    model = torch.load(filename, map_location=torch.device('cpu'))

    # Create corresponding .pkl filename
    pkl_filename = os.path.splitext(filename)[0] + '.pkl'

    # Save the model as .pkl file
    with open(pkl_filename, 'wb') as f:
        pickle.dump(model, f)
