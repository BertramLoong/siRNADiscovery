from Bio import SeqIO
import pandas as pd
import subprocess


# Read mrna file
mrna_file = "mRNA.fas"
mrna_map = []
for seq_record in SeqIO.parse(mrna_file,"fasta"):
	mrna_id = seq_record.id
	seq = seq_record.seq
	mrna_map.append((mrna_id, str(seq)))

mrna_map = pd.DataFrame(mrna_map, columns = ['mRNA', 'mRNA_seq'])


# Read sirna file
sirna_file = "siRNA.txt"
sirna_map = []
for seq_record in SeqIO.parse(sirna_file, "fasta"):
	sirna_id = seq_record.id
	seq = seq_record.seq
	seq = seq.upper()
	sirna_map.append((sirna_id,str(seq)))

sirna_map = pd.DataFrame(sirna_map, columns = ['siRNA','siRNA_seq'])

# Combine mRNA & siRNA file
efiicacy_file = pd.read_csv("sirna_mrna_efficacy.csv")
eff_df = efiicacy_file.merge(sirna_map, on = "siRNA")
eff_df = eff_df.merge(mrna_map, on = "mRNA")


# Construct seq-pairs
eff_df['seq_pairs'] = eff_df['siRNA_seq']+'&'+eff_df['mRNA_seq']


# RNAcofold
for ind, row in eff_df.iterrows():
	pairs = row[4]

	id_name = row[0] + "_" + row[1]
	print (id_name)

	proc = subprocess.Popen(['/usr/local/bin/RNAcofold','-p',"--id-prefix=" + id_name], 
		stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)

	output, error = proc.communicate(pairs)
	# print (output)

print ("finished!")








