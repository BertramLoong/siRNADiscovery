mrna_fasta_file = 'mRNA.fas'
sirna_fasta_file = 'siRNA.txt'


if __name__ == '__main__':
    with open(mrna_fasta_file, 'r') as file:
        title = file.readline().split(">")[1].strip()

        mrna_seq = ''

        for line in file:
            mrna_seq += line.strip()

        processed_seq = ''
        for char in mrna_seq:
            if char in ['A', 'T', 'C', 'G']:
                processed_seq += char.upper()
            else:
                processed_seq += char.lower()   

    mrna_len = len(processed_seq)
    sirna_dict = {}
    for i in range(mrna_len - 20):
        si_name = '>test_' + str(i)
        si_seq = mrna_seq[i: i + 21]
        si_seq = si_seq.lower()
        si_seq = si_seq[:: -1]
        si_seq = si_seq.replace('a', 'U').replace('t', 'A').replace('g', 'C').replace('c', 'G')
        sirna_dict[si_name] = si_seq

    with open(sirna_fasta_file, 'w') as file:
        for key in sirna_dict.keys():
            file.write(key + '\n')
            file.write(sirna_dict[key] + '\n')
