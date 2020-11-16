#import datetime
#import os
import h5py
import numpy as np

#with h5py.File('rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5') as f:
with h5py.File('rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5', 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])