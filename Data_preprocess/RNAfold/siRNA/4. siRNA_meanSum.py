import numpy as np
import pandas as pd
import os

path = "RNAfold_reduced_matrix6"
files = os.listdir(path)

df = []
first_parts = []

for file in files:
    file_name = file.replace('_0001.npy', '')
    parts = file_name.split('_0001', 1)
    first_parts.append(parts[0])

    data = np.load(os.path.join(path, file))
    result_data = data.mean(0)

    df.append(result_data)

df_res = pd.DataFrame(df)
df_res.index = first_parts
print(df_res)

df_res.to_csv("self_siRNA_matrix_meanSum6.txt", header=False)
