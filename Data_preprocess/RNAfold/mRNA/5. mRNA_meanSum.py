import numpy as np
import pandas as pd
import os


path = "RNAfold_reduced_matrix100"
files = os.listdir(path)


df = []
first_parts = []

for file in files:
    file_name = file.replace('_0001.npy', '')

    parts = file_name.split('_0001', 1)
    first_parts.append(parts[0])

    data = np.load(path + "/" + file)
    data = data.mean(0)
    df.append(data)


df_res = pd.DataFrame(df)
df_res.index = first_parts
print(df_res)


df_res.to_csv("self_mRNA_matrix_meanSum100.txt",header = False)



