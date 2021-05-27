#Adaptacion posicion con variacion de cancer (localizacion 39..10,377):
standard_table = CodonTable.unambiguous_dna_by_id[1]
#transformer
#se incialia el modelo sin caso alguno
configuration = BertConfig()
#se construye un modelo con el caso base sin parametros
model = BertModel(configuration)
#hay que configurar el modelo mediante
configuration = model.config
#primera vista del modelo de transfomer para el tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(rna,return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
#las prediccione seran para crear una clasificacion
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
#tokens para la clasificacion
inputs = tokenizer(rna, return_tensors="pt")
labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits