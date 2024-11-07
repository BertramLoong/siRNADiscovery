# siRNADiscovery: A Graph Neural Network for siRNA Efficacy Prediction via Deep RNA Sequence Analysis

This repository contains the source code for **siRNADiscovery**.

Rongzhuo Long†, Ziyu Guo†, Da Han, Boxiang Liu, Xudong Yuan*, Guangyong Chen*, Pheng-Ann Heng, Liang Zhang*<sup>#</sup>, siRNADiscovery: a graph neural network for siRNA efficacy prediction via deep RNA sequence analysis, *Briefings in Bioinformatics*, Volume 25, Issue 6, November 2024, bbae563, https://doi.org/10.1093/bib/bbae563

† contributed equally, 
\* corresponding authors, 
<sup>#</sup> lead corresponding author

For questions or further information, please contact the lead corresponding author, Liang Zhang, at [zhangliang@him.cas.cn](mailto:zhangliang@him.cas.cn).

## Table of Contents

- [siRNADiscovery: A Graph Neural Network for siRNA Efficacy Prediction via Deep RNA Sequence Analysis](#sirnadiscovery-a-graph-neural-network-for-sirna-efficacy-prediction-via-deep-rna-sequence-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Data Preprocessing](#data-preprocessing)
    - [RNA-Protein Interaction Probabilities](#rna-protein-interaction-probabilities)
    - [siRNA-mRNA Base-Pairing Probabilities](#sirna-mrna-base-pairing-probabilities)
  - [Dependencies](#dependencies)
  - [Running the Code](#running-the-code)

## Overview

siRNADiscovery leverages a graph neural network (GNN) to predict the efficacy of small interfering RNA (siRNA) sequences in gene silencing. This model incorporates RNA-protein interaction probabilities and siRNA-mRNA base-pairing probabilities as part of its predictive framework.

## Data Preprocessing

### RNA-Protein Interaction Probabilities

To generate RNA-protein interaction probabilities, we utilize tools provided by the [RPISeq website](http://pridb.gdcb.iastate.edu/RPISeq/). These probabilities are essential for building RNA-protein interaction features used in siRNADiscovery.

### siRNA-mRNA Base-Pairing Probabilities

For siRNA-mRNA base-pairing probabilities, we uss the [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/). Specifically, tools like **RNAcofold** and **RNAfold** are employed for calculating these pairing probabilities.

Refer to the **Data_preprocess** folder for detailed instructions on executing RNAcofold and RNAfold.


## Dependencies

Ensure that you have the following installed to run the code smoothly:

- Python 3.x
- Required Python packages (list of dependencies in `requirements.txt`)
- ViennaRNA (for siRNA-mRNA base-pairing calculations)

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

## Running the Code

You can reproduce the results by executing the relevant scripts located in the `siRNA_split` and `mRNA_split` folders.

To run the model, use the following command:

```bash
python siRNADiscovery.py
```
