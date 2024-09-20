import numpy as np
import pandas as pd
import os


# reduced_matrix
path = "RNAcofold_reduced_matrix50"
files = os.listdir(path)

df = []
first_parts = [] 
remaining_parts = []

for file in files:
    file_name = file.replace('_0001.npy', '')

    parts = file_name.split('_', 1)
    first_parts.append(parts[0])
    remaining_parts.append(parts[1])


    data = np.load(path + "/" + file)
    data = data.mean(0) 
    df.append(data)

df_res = pd.DataFrame(df)

df_res.insert(0, 'siRNA', first_parts)
df_res.insert(1, 'mRNA', remaining_parts)

df_res['index'] = df_res['siRNA'] + '_' + df_res['mRNA']
df_res.set_index('index', inplace=True)
df_res.drop(['siRNA', 'mRNA'], axis=1, inplace=True)
print(df_res.head(3))


df_res.to_csv("con_matrix_meanSum50.txt",header = False)




