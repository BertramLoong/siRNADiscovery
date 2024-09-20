import glob
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import save_npz
import re
import os


max_sirna_len = 21
max_mrna_len = 9756
matrix_size = max_sirna_len + max_mrna_len + 2
n_components = 50

file_paths = glob.glob('RNAcofold_bp_file/*dp.ps.bpp')

with cp.cuda.Device(1):
    for file_path in file_paths:
        pos_data = np.loadtxt(file_path, usecols=[0, 1, 2])
        pos_data_gpu = cp.asarray(pos_data)


        zero_matrix = cp.zeros((matrix_size, matrix_size), dtype=cp.float32)
        zero_matrix[pos_data_gpu[:, 0].astype(cp.int32) - 1, pos_data_gpu[:, 1].astype(cp.int32) - 1] = pos_data_gpu[:, 2]

        mask = cp.ones(matrix_size, dtype=cp.bool_)
        mask[max_sirna_len:max_sirna_len + 2] = False
        pos_matrix = zero_matrix[mask][:, mask]
        pos_matrix = pos_matrix + pos_matrix.T - cp.diag(cp.diag(pos_matrix))

        data_matrix_cpu = cp.asnumpy(pos_matrix)
        sparse_matrix = csr_matrix(data_matrix_cpu)

        
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        reduced_data = svd.fit_transform(sparse_matrix)

        directory_path = 'RNAcofold_reduced_matrix'+ str(n_components)
        
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
            
        np.save('RNAcofold_reduced_matrix'+ str(n_components) +'/'+file_path.split('/')[-1].split('_dp.ps.bpp')[0]+'.npy',reduced_data)


        print (file_path + " finished!")

