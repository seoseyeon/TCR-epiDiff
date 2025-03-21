import os
import numpy as np
import pandas as pd
import pickle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

#### CDR3b One-hot Encoding ####
# Mapping of amino acids to their representative codons
amino_acid_to_codon = {
    'A': 'GCC', 'C': 'TGC', 'D': 'GAC', 'E': 'GAG',
    'F': 'TTC', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC',
    'K': 'AAG', 'L': 'CTG', 'M': 'ATG', 'N': 'AAC',
    'P': 'CCC', 'Q': 'CAG', 'R': 'AGA', 'S': 'AGC',
    'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAC', '*': "TGA" 
}

def amino_acids_to_codons(sequence):
    seq = [amino_acid_to_codon[amino_acid] for amino_acid in sequence if amino_acid in amino_acid_to_codon]
    seq_string = ''.join(seq)
    return seq_string

def nuclei_acids_to_one_hot(sequences, target_length=72):
    one_hot_encoded = []
    
    for sequence in sequences:
        # Convert amino acid sequence to codon sequence
        codon_seq = ''.join(amino_acid_to_codon[aa] for aa in sequence if aa in amino_acid_to_codon)
        
        # Initialize one-hot encoding array (4, target_length)
        encoding = np.zeros((4, target_length), dtype=int)
        
        # Convert nucleotide sequence to one-hot encoding
        for i, nucleotide in enumerate(codon_seq):
            if nucleotide in nucleotide_to_index and i < target_length:
                encoding[nucleotide_to_index[nucleotide], i] = 1
        
        one_hot_encoded.append(encoding)
    
    return one_hot_encoded

#### Epitope Embedding ####
# Load ProtT5 model & T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
model = T5ForConditionalGeneration.from_pretrained('Rostlab/prot_t5_xl_uniref50')

class ProteinEmbeddings:
    def __init__(self, sequences):
        self.sequences = sequences
        self.embeddings = self.get_protein_embeddings(sequences)

    def preprocess_sequence(self, sequence):
        return " ".join(sequence)

    def get_protein_embeddings(self, sequences): 
        processed_sequences = [self.preprocess_sequence(seq) for seq in sequences]
        inputs = tokenizer(
            processed_sequences, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        input_ids = inputs.input_ids

        # Encoder 
        with torch.no_grad():
            encoder_outputs = model.encoder(input_ids)
        
        # Extraction Last Hidden State -> Pooling
        embeddings = encoder_outputs.last_hidden_state.mean(dim=1)

        return embeddings

    def get_embedding_dict(self):
        return {seq: embedding.numpy() for seq, embedding in zip(self.sequences, self.embeddings)}


#### HLA one-hot encoding ####
# HLA-A* = [1, 0, 0], HLA-B* = [0, 1, 0], HLA-C* = [0, 0, 1]

def encode_hla(hla):
    if hla.startswith("A"):
        return [1, 0, 0]
    elif hla.startswith("B"):
        return [0, 1, 0]
    elif hla.startswith("C"):
        return [0, 0, 1]



#### Data processing ####
# Load Data
Data = pd.read_csv("/data.csv")
Data = Data[["CDR3", "Epitope", "MHC A"]]
Data = Data[Data["CDR3"].str.len().between(7, 24)] # 7<=CDR3b<=24
Data = Data[~Data.iloc[:, -1].str.contains(r'[DE]', na=False)] # only HLA-A/B/C

# Convert Nuclei Acids to One-hot encoding
codon_sequences = []
for seq in cdr3b:
    codon_sequence = amino_acids_to_codons(seq)
    codon_sequences.append(codon_sequence)
    
one_hot_results = nuclei_acids_to_one_hot(codon_sequences)
Data["TCR_one_hot"] = one_hot_results

# Protein embeddings
protein_embeddings = ProteinEmbeddings(Data.Epitope.Unique())

# Dictionary
protein_embedding_dict = protein_embeddings.get_embedding_dict()
Data['Epitope_embedding'] = Data['Epitope'].map(protein_embedding_dict)

# HLA one-hot encodding
Data["HLA encoding"] = Data["new_HLA"].apply(encode_hla)

result = []

for tcr_onehot, epitope_embedding, HLA_embedding, real_TCR, real_epitope, real_HLA in zip(
    list(Data["TCR_one_hot"]), 
    list(Data["Epitope_embedding"]),
    list(Data["HLA encoding"]), 
    list(Data["CDR3"]), 
    list(Data["Epitope"]),
    list(Data["new_HLA"])
):
    tcr_onehot = torch.tensor(tcr_onehot, dtype=torch.float32)
    epitope_embedding = torch.tensor(epitope_embedding, dtype=torch.float32)
    HLA_embedding = torch.tensor(HLA_embedding, dtype=torch.float32) 
    
    result.append((tcr_onehot, epitope_embedding, HLA_embedding, torch.tensor(1, dtype=torch.float32), (real_TCR), (real_epitope), (real_HLA))) 
    # If you want to create negative samples, use this code
    #result.append((tcr_onehot, epitope_embedding, HLA_embedding, torch.tensor(0, dtype=torch.float32), (real_TCR), (real_epitope), (real_HLA)))

# Save
with open("/DiffBP_trainig_final_pos.pkl", "wb") as file:
    pickle.dump(result, file)


