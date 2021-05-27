#primer paso sera hacer la lectura de archivos
from Bio import *
from numpy import *
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import translate, transcribe, back_transcribe, Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Data import CodonTable
import re
#from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipline
#from transformers import BertModel, BertConfig, BertTokenizer, BertModel, BertForSequenceClassification
#import torch
f=open('segCancer.fasta','r')
seqADN=f.read()
f.close() #val al final
print("\nInformacion de la cadena en formato ADN:\n")
print(seqADN)
#hay que secuenciar la informacion para el transformer
seq=Seq(seqADN)
#aqui hay que hacer el cambio a AN y luego ARN
rna=seq.transcribe()
str(rna)
print("\nInformacion del ADN en forma de ARN:\n")
print(rna)
arc=str(rna)
q=open('inEmbSegCancer.txt','w')
q.write(arc)
q.close()
#expresion regualar pa hacer el split aca chido
def test_patterns(text,patterns):
	text=arc
	patterns=["AUG","UGA","UAA"]

	for pattern desc in patterns:
		print("'{}'({})\n".format(pattern,desc))
		print("    '{}'".format(text))
		for match in re.finditer(pattern,text):
			s=match.start()
			e=match.end()
			substr=text[s:e]
			n_backslashes=text[:s].count('\\')
			prefix='.'*(s+n_backslashes)
			print("   {}'{}'".format(prefix,substr))
		print()
	return

test_patterns('AUGAAAAAUGAUAAAAAGGGCCC','AUG',"'A' followed by 'UG'")


