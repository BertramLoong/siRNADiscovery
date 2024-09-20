from Bio import SeqIO
import numpy
import pandas
import math

# sirna position scores
position_scores = [
    {"A": -1, "C": 1, "G": 1, "U": -1},
    {"A": -1, "C": 0, "G": 1, "U": -1},
    {"A": 1, "C": -1, "G": 1, "U": -1},
    {"A": 0, "C": -1, "G": 0, "U": 1},
    {"A": 1, "C": 0, "G": 0, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": 1, "U": -1},
    {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 0, "C": 0, "G": -1, "U": -1},
    {"A": 1, "C": 1, "G": 1, "U": 1},
    {"A": 0, "C": 1, "G": 1, "U": 0},
    {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 0, "C": -1, "G": 0, "U": 0},
    {"A": 1, "C": -1, "G": 0, "U": -1},
    {"A": 0, "C": 0, "G": 1, "U": 1},
    {"A": 1, "C": 0, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1}
]


# Transfer seq to one-hot
def obtain_one_hot_feature_for_one_sequence_1(seq1, max_len):
    mapping = dict(zip("NACGT", range(5)))
    seq2 = [mapping.get(i, 0) for i in seq1]

    zero_arr = numpy.zeros((max_len - len(seq1), 4), dtype=numpy.uint8) 
    unit_arr = numpy.concatenate((numpy.zeros((1, 4), dtype=numpy.uint8), numpy.eye(4, dtype=numpy.uint8)))

    seq2 = unit_arr[seq2]
    return numpy.concatenate((seq2, zero_arr)).flatten()



# Positional encoding

def get_pos_embedding(index, d_model, t):
    if index % 2 == 0:
        return math.sin(t / pow(10000, index / (d_model - 1)))
    else:
        return math.cos(t / pow(10000, index / (d_model - 1)))


def get_pos_embedding_sequence(mrna_start_pos, len_sirna, d_model):
    pe_list = []
    for offset in range(len_sirna):
        t = mrna_start_pos + offset
        for i in range(d_model):
            pe = get_pos_embedding(i, d_model, t)
            pe = round(pe, 4)
            pe_list.append(pe)
    return pe_list


# Thermodynamics
def build_kmers(sequence):
    kmers = []
    n_kmers = len(sequence) - 1

    for i in range(n_kmers):
        kmer = sequence[i:i + 2]
        kmers.append(kmer)
    return kmers

def cal_thermo_feature(sequence,intermolecular_initiation=4.09, symmetry_correction=0.43):
    sum_stability = 0
    single_sum = []

    bimers = build_kmers(sequence)

    if sequence[0] == 'A':
        sum_stability += 0.45
    if sequence[18] == 'U': 
        sum_stability += 0.45

    for b in bimers:
        bimer_values = {
            'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33,
            'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11,
            'GU': -2.24, 'AC': -2.24, 'GA': -2.35, 'UC': -2.35,
            'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42
        }
        stability_value = bimer_values.get(b, 0)
        single_sum.append(stability_value)
        sum_stability += stability_value

    sum_stability += intermolecular_initiation
    sum_stability += symmetry_correction
    single_sum.append(round(sum_stability,2))

    return single_sum

# count GC percentage
def countGC(seq):
    seq = seq.upper()
    gc_percent = (seq.count("G")+seq.count("C"))/len(seq)
    return round(gc_percent,3)


# count k-mers
## single base
def single_freq(seq):
    seq = seq.upper().replace("T","U")
    freq = {base: seq.count(base) / len(seq) for base in "ACGU"}

    return freq


## double base
def double_freq(seq):
    seq = seq.upper().replace("T","U")
    all_double_base = [a + b for a in "ACGU" for b in "ACGU"]

    double_base_freq = {dbase: 0 for dbase in all_double_base}
    totabl_base = len(seq) - 1 

    for i in range(totabl_base):
        double_base = seq[i:i+2]
        double_base_freq[double_base] += 1 

    double_base_freq = {key : value / totabl_base for key, value in double_base_freq.items()}

    return double_base_freq



## triple base
def triple_freq(seq):
    seq = seq.upper().replace("T","U")

    all_triple_base = [a + b + c for a in "ACGU" for b in "ACGU" for c in "ACGU"]

    triple_base_freq = {tbase: 0 for tbase in all_triple_base} 
    totabl_base = len(seq) - 2 

    for i in range(totabl_base):
        triple_base = seq[i:i+3]
        triple_base_freq[triple_base] += 1 

    triple_base_freq = {key : value / totabl_base for key, value in triple_base_freq.items()}

    return triple_base_freq


## quadruple base
def quadruple_freq(seq):
    seq = seq.upper().replace("T","U")

    all_quadruple_base = [a + b + c + d for a in "ACGU" for b in "ACGU" for c in "ACGU" for d in "ACGU"]

    quadruple_base_freq = {tbase: 0 for tbase in all_quadruple_base} 
    totabl_base = len(seq) - 3 

    for i in range(totabl_base):
        quadruple_base = seq[i:i+4]
        quadruple_base_freq[quadruple_base] += 1 

    quadruple_base_freq = {key : value / totabl_base for key, value in quadruple_base_freq.items()}

    return quadruple_base_freq


## quintuple base
def quintuple_freq(seq):
    seq = seq.upper().replace("T","U")

    all_quintuple_base = [a + b + c + d + e for a in "ACGU" for b in "ACGU" for c in "ACGU" for d in "ACGU" for e in "ACGU"]

    quintuple_base_freq = {tbase: 0 for tbase in all_quintuple_base} 
    totabl_base = len(seq) - 4 

    for i in range(totabl_base):
        quintuple_base = seq[i:i+5]
        quintuple_base_freq[quintuple_base] += 1 

    quintuple_base_freq = {key : value / totabl_base for key, value in quintuple_base_freq.items()}

    return quintuple_base_freq


# Calculate siRNA rules codes
def one_hot_encode(score):
    """One-hot encode the score (-1, 0, 1)."""
    if score == -1:
        return [1, 0, 0]
    elif score == 0:
        return [0, 1, 0]
    elif score == 1:
        return [0, 0, 1]

def rules_scores(seq):
    """ Obtain siRNA each position scores with one-hot encoding.

    Args:
        seq (str): siRNA sequence

    Returns:
        list: siRNA each position scores of ahead 19nt with one-hot encoding
    """
    seq = seq.upper().replace("T","U")
    one_hot_scores = []
    for i in range(19):
        score = position_scores[i][seq[i]]
        one_hot_scores.extend(one_hot_encode(score))
    return one_hot_scores





























