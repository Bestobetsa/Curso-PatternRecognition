

w=open('bagOfWord.fasta','r')
ref=w.read()
w.close()
print("\nInformacion de la variacion de cancer:\n")
seq1=Seq(ref)
rna1=seq1.translate(table=1)
rnaRef=seq1.transcribe()
str(rnaRef)
print(rnaRef)
