from Bio import SeqIO
import pandas as pd
import subprocess


# Read mRNA file
mrna_file = "mRNA.fas"
mrna_map = []
for seq_record in SeqIO.parse(mrna_file,"fasta"):
	mrna_id = seq_record.id
	seq = seq_record.seq
	mrna_map.append((mrna_id, str(seq)))

mrna_map = pd.DataFrame(mrna_map, columns = ['mRNA', 'mRNA_seq'])


# RNAcofold
for ind, row in mrna_map.iterrows():
	seq = row[1]
	
	id_name = row[0]

	print (id_name)

	proc = subprocess.Popen(['/usr/local/bin/RNAfold','-p',"--id-prefix=" + id_name], 
		stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)

	output, error = proc.communicate(seq)

print ("finished !")












