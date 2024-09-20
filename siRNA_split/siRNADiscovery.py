import stellargraph as StellarGraph
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import re, json

from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE
from tensorflow.keras import layers, Model, optimizers,callbacks
from sklearn.metrics import roc_auc_score

import utils # Features calculation

params = json.load(open("siRNA_param.json", 'r'))



score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []



for n in range(10):

    """

    Read File

    """

    # Read file
    data_train = pd.read_csv("siRNA_split_datasets/split" + str(n) + "/train.csv")
    data_dev = pd.read_csv("siRNA_split_datasets/split" + str(n) + "/dev.csv")
    data_test = pd.read_csv("siRNA_split_datasets/split" + str(n) + "/test.csv")


    # Substitue U to T
    data_train['siRNA_seq'] = data_train['siRNA_seq'].replace('U', 'T', regex=True)
    data_dev['siRNA_seq'] = data_dev['siRNA_seq'].replace('U', 'T', regex=True)
    data_test['siRNA_seq'] = data_test['siRNA_seq'].replace('U', 'T', regex=True)


    # Concat data
    data = pd.concat([data_train,data_dev,data_test],axis=0)


    '''

    Feature processing

    '''

    # one-hot
    ## siRNA
    sirna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["sirna_length"]) for seq in data['siRNA_seq']]
    sirna_onehot = pd.DataFrame(sirna_onehot,index=list(data['siRNA']))


    ## mRNA
    mrna_onehot_temp = data.loc[:,['mRNA','mRNA_seq']]
    mrna_onehot_temp = mrna_onehot_temp.drop_duplicates(subset="mRNA")

    mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["max_mrna_len"]) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_onehot = pd.DataFrame(mrna_onehot,index = list(mrna_onehot_temp['mRNA']))


    # Positional encoding
    trans_table = str.maketrans('ATCG', 'TAGC')
    data['match_pos'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
    data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].index(row['match_pos']),axis = 1)


    sirna_pos_encoding = [utils.get_pos_embedding_sequence(num,params["sirna_length"],params["dmodel"]) for num in data['match_pos']]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding,index = list(data['siRNA']))


    # Thermodynamics
    sirna_thermo_feat = [utils.cal_thermo_feature(seq.replace("T","U")) for seq in data['siRNA_seq']]

    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    sirna_thermo_feat = pd.concat([data['siRNA'].reset_index(drop=True),
                                   data['mRNA'].reset_index(drop=True),
                                   sirna_thermo_feat],
                                   axis = 1)

    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])


    # Co-fold features
    con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt",header=None,index_col=0)

    con_feat = con_feat.reindex(sirna_thermo_feat.index)


    # sel-fold features
    ## siRNA
    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt",header=None,index_col=0)

    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)

    ## mRNA
    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt",header=None,index_col=0)

    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index)


    # AGO2
    ## siRNA-AGO2
    sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2.csv",index_col = 0)

    sirna_ago = sirna_ago.reindex(sirna_onehot.index)

    ## mRNA-AGO2
    mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2.csv",index_col=0)

    mrna_ago = mrna_ago.reindex(mrna_onehot.index)


    # GC percentage
    ## siRNA
    sirna_GC = [utils.countGC(seq) for seq in data['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC,index=list(data['siRNA']))

    ## mRNA
    mrna_GC = [utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_onehot_temp['mRNA']))

    # k-mers
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

    sirna_k_mers = pd.concat([sirna_1_mer,sirna_2_mers, sirna_3_mers,sirna_4_mers,sirna_5_mers], axis = 1)
    sirna_k_mers.index = data['siRNA']


    # siRNA rules codes
    sirna_pos_scores = [utils.rules_scores(seq) for seq in data['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index = list(data['siRNA']))


    '''

    The features of GNN nodes

    '''

    # siRNA nodes
    sirna_pd = pd.concat([sirna_onehot,sirna_sfold_feat,sirna_ago,sirna_GC,sirna_k_mers,sirna_pos_scores],axis = 1)


    # mRNA nodes
    mrna_pd = pd.concat([mrna_onehot,mrna_sfold_feat,mrna_ago,mrna_GC],axis = 1)


    # edges
    source = data['siRNA'] + "_" + data['mRNA']
    target_siRNA = data['siRNA']
    target_mRNA = data['mRNA']

    all_my_edges1 = pd.DataFrame({'source':source,'target':target_siRNA})
    all_my_edges2 = pd.DataFrame({'source':source,'target':target_mRNA})

    all_my_edges = pd.concat([all_my_edges1,all_my_edges2],ignore_index=True, axis=0)


    # interactive nodes
    sirna_pos_encoding.index = sirna_thermo_feat.index

    interaction_pd = pd.concat([sirna_thermo_feat,con_feat,sirna_pos_encoding],axis=1)


    '''

    Model training

    '''

    # Stellargraph object
    my_stellar_graph = StellarGraph.StellarGraph({"siRNA": sirna_pd, "mRNA": mrna_pd, "interaction": interaction_pd},
                                             edges = all_my_edges, source_column="source", target_column="target")

    # Generator
    generator = HinSAGENodeGenerator(my_stellar_graph, params["batch_size"], params["hop_samples"], head_node_type="interaction")

    hinsage_model = HinSAGE(
            layer_sizes = params["hinsage_layer_sizes"], generator=generator, bias=True, dropout = params["dropout"]
        )

    x_inp, x_out = hinsage_model.in_out_tensors()

    # Predictor
    prediction = layers.Dense(units=1)(x_out)

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr = params["lr"]),
        loss = params["loss"]
    )

    # train flow
    train_interaction = pd.DataFrame(data_train['efficacy'].values, index=data_train['siRNA'] + "_" + data_train['mRNA'])
    train_gen = generator.flow(train_interaction.index, train_interaction, shuffle=True)


    # dev flow
    dev_interaction = pd.DataFrame(data_dev['efficacy'].values, index=data_dev['siRNA'] + "_" + data_dev['mRNA'])

    dev_gen = generator.flow(dev_interaction.index, dev_interaction)


    # model fit
    history = model.fit(train_gen, epochs= params["epochs"], validation_data=dev_gen, verbose=2, shuffle=False)


    # Plot the training history
    history_plot = StellarGraph.utils.plot_history(history, return_figure=True)


    '''

    Evaluate on the testset

    '''

    # test flow
    test_interaction = pd.DataFrame(data_test['efficacy'].values, index=data_test['siRNA'] + "_" + data_test['mRNA'])
    test_gen = generator.flow(test_interaction.index, test_interaction)

    # predict
    test = model.predict(test_gen)
    test = np.squeeze(test)


    '''

    Calculate the metrics

    '''

    # PCC
    pearson = scipy.stats.pearsonr(test_interaction.values.ravel(), test)
    score_PCC.append(pearson[0])
    print(pearson[0])

    # SPCC
    spearman = scipy.stats.spearmanr(test_interaction.values.ravel(), test)
    score_SPCC.append(spearman[0])
    print(spearman[0])

    # MSE
    mse_run = mean_squared_error(test_interaction.values.ravel(), test)
    score_mse.append(mse_run)
    print(mse_run)

    # AUC
    binary_pred = (test_interaction.values.ravel() > 0.7).astype(int)
    auc = roc_auc_score(binary_pred, test)
    score_auc.append(auc)
    print (auc)


    print (str(n)+ " " + "finished!")


# Print the average value of metrics
print("Overall PCC score = ", np.mean(score_PCC))
print("Overall SPCC score = ", np.mean(score_SPCC))
print("Overall MSE score = ", np.mean(score_mse))
print("Overall AUC score = ", np.mean(score_auc))