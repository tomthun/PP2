NPZ files can be loaded as follows:

import numpy as np
from pathlib import Path

npz_path = Path("/path/to/data.npz") # path to data
npz_data = np.load( npz_path, mmap_mode='r' ) # load data
# create dictionary with protein identiifers as keys and embeddings as values
# embeddings are a matrix of size Lx1024 (=standard SeqVec, see paper) or Lx64 (=autoencoded SeqVec; indicated by files ending with '_64dims.npz')
# the latter is derived from reducing 3076 dimensions (raw SeqVec output, not summed yet) to 64 via autoencoder bottleneck
# values are numpy matrices
npz_data = dict( npz_data )

