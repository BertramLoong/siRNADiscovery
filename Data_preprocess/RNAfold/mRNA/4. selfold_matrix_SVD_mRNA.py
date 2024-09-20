import glob
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import save_npz
import re
import os


matrix_size = 9756
n_components = 100

file_paths = glob.glob('RNAfold_bp_file/*dp.ps.bpp')

with cp.cuda.Device(0): 
    for file_path in file_paths:
        pos_data = np.loadtxt(file_path, usecols=[0, 1, 2])
        pos_data_gpu = cp.asarray(pos_data) 


        pos_matrix = cp.zeros((matrix_size, matrix_size), dtype=cp.float32) 
        pos_matrix[pos_data_gpu[:, 0].astype(cp.int32) - 1, pos_data_gpu[:, 1].astype(cp.int32) - 1] = pos_data_gpu[:, 2]
        pos_matrix = pos_matrix + pos_matrix.T - cp.diag(cp.diag(pos_matrix))
      
      
        data_matrix_cpu = cp.asnumpy(pos_matrix)
        sparse_matrix = csr_matrix(data_matrix_cpu)

        
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        reduced_data = svd.fit_transform(sparse_matrix)

        directory_path = 'RNAfold_reduced_matrix'+ str(n_components)

        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

        np.save('RNAfold_reduced_matrix'+ str(n_components) + '/' + file_path.split('/')[-1].split('_dp.ps.bpp')[0]+'.npy',reduced_data)


        print (file_path + " finished!")

print ("finished!")