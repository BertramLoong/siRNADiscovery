from Bio import SeqIO
import pandas as pd
import subprocess



# Read sirna file
sirna_file = "siRNA.txt"
sirna_map = []
for seq_record in SeqIO.parse(sirna_file, "fasta"):
	sirna_id = seq_record.id
	seq = seq_record.seq
	seq = seq.upper()
	sirna_map.append((sirna_id,str(seq)))

sirna_map = pd.DataFrame(sirna_map, columns = ['siRNA','siRNA_seq'])


# RNAcofold
for ind, row in sirna_map.iterrows():
	seq = row[1]
	
	id_name = row[0]
	
	proc = subprocess.Popen(['/usr/local/bin/RNAfold','-p',"--id-prefix=" + id_name], 
		stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)

	output, error = proc.communicate(seq)

print ("finished !")














