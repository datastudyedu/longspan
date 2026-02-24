# Disclaimer: Generativ AI have been utilized to create the longevity DNA editing program.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# DNA encoding function
def encode_dna(sequence, nucleotide_to_int, max_length):
    encoded = [nucleotide_to_int.get(nuc, 0) for nuc in sequence]
    return encoded + [0] * (max_length - len(encoded))  # Pad to max_length

# Model to predict lifespan based on DNA sequence
class DNA_Lifespan_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_length):
        super(DNA_Lifespan_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding='same')
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Reshape for convolution
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define nucleotide-to-integer mapping (for DNA encoding)
nucleotide_to_int = {
    'A': 1,
    'C': 2,
    'G': 3,
    'T': 4,
    'a': 1,
    'c': 2,
    'g': 3,
    't': 4
}

# Train model function
def train_model(X_train, y_train):
    # Set up the model, loss function, and optimizer
    vocab_size = len(nucleotide_to_int) + 1  # Including padding value 0
    embedding_dim = 8
    input_length = 1000
    model = DNA_Lifespan_Model(vocab_size, embedding_dim, input_length)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # DataLoader setup
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Training loop
    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
    
    return model

# Function to simulate mutation in the DNA sequence
def mutate_dna(dna_sequence, mutation_rate=0.01):
    # Introduce random mutations with a given probability
    mutated_sequence = list(dna_sequence)
    for i in range(len(mutated_sequence)):
        if np.random.rand() < mutation_rate:
            mutated_sequence[i] = np.random.choice(['A', 'C', 'G', 'T'])  # Randomly mutate to any nucleotide
    return ''.join(mutated_sequence)

# Function to predict lifespan based on DNA sequence
def predict_lifespan(model, dna_sequence, nucleotide_to_int, max_length=1000):
    encoded_sequence = encode_dna(dna_sequence, nucleotide_to_int, max_length)
    padded_seq_tensor = torch.tensor([encoded_sequence], dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        predicted_lifespan = model(padded_seq_tensor).item()
    
    return predicted_lifespan

# Function to optimize lifespan by mutation
def optimize_lifespan(model, initial_sequence, nucleotide_to_int, mutation_rate=0.05, max_iterations=10000):
    best_sequence = initial_sequence
    best_lifespan = predict_lifespan(model, initial_sequence, nucleotide_to_int)

    for iteration in range(max_iterations):
        mutated_sequence = mutate_dna(best_sequence, mutation_rate)
        mutated_lifespan = predict_lifespan(model, mutated_sequence, nucleotide_to_int)
        
        if mutated_lifespan > best_lifespan:
            best_lifespan = mutated_lifespan
            best_sequence = mutated_sequence
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best lifespan {best_lifespan:.2f} years")
    
    return best_sequence, best_lifespan

# Example DNA sequences and labels (for demonstration purposes)
# Encoding some random DNA sequences for training
sample_sequences = [
    ('gcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagc' +
    'ctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcct' +
    'aagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaa' +
    'gcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagc' +
    'ctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcctaagcct' +
    'aagcctaagcctaagcctaagcctaagcctaaaaaattgagataagaaaacattttactttttcaaaattgttttcatgc' +
    'taaattcaaaacgttttttttttagtgaagCTTCTAGATATTTGGCGGGTACCTCTAATTTTGCCTGCCTGCCAACCTAT').upper(),
    ('ATGCTCCTGTGTTTAGGCCTAatactaagcctaagcctaagcctaatactaagcctaagcctaagactaagcctaatact' +
    'aagcctaagcctaagactaagcctaagactaagcctaagactaagcctaatactaagcctaagcctaagactaagcctaa' +
    'gcctaatactaagcctaagcctaagactaagcctaatactaagcctaagcctaagactaagcctaagactaagcctaaga' +
    'ctaagcctaatactaagcctaagcctaagactaagcctaagcctaaaagaATATGGTAGCTACAGAAACGGTAGTACAct' +
    'cttctgaaaatacaaaaaatttgcaatttttatagctaGGGCACTTTTTGTCTGCCCAAATATAggcaaccaaaaataat' +
    'tgccaagtttttaatgatttgttgcatattgaaaaaaacatttttcgggttttttgaaatgaatatcgtAGCTACAGAAA' +
    'CGGTTGTGCACTCATCTgaaagtttgtttttcttgttttcttgcACTTTGTGCAGAATTCTTGATTCTTGATTCTTGCAg' +
    'aaatttgcaagaaaattcgcaagaaatttgtattaaaaactgttcaaaatttttggaaattagtttaaaaatctcacatt').upper(),
    ('ttttttagaaaaattatttttaagaatttttcattttaggaatattgttatttcagaaaatagctAAATGTGATTTCTGT' +
    'AATTTTGCCTGCCAAATTCGTGAAATGCAATAAAAATCTAATATCCCTCATCAGTGCGATTTCCGAATCAGTATATTTTT' +
    'ACGTAATAGCTTCTTTGACATCAATAAGTATTTGCCTATATGACTTTAGACTTGAAATTGGCTATTAATGCCAATTTCAT' +
    'GATATCTAGCCACTTTAGTATaattgtttttagtttttggcaaaactatTGTCTAAACAGATATTCgtgttttcaagaaa' +
    'tttttcatggtttttctTGGTCTTTTCTTggtatttttttgacaaaaatttttgtttcttgattcttgcaaaaatttttc' +
    'cgtttgACGGCCTTGATGTGCACTACCTTCGCTTAAATactacattttctgaaaatgttataatAGTGTTCATTGTTTCA' +
    'TACAAATACTTATTTAATAGTATTTCTGGTTATATAATTTGTATAAAAAGTGGTTGACATAACAAGGCTGACGAAACTTT' +
    'GTgatggctgaaaatattttcctagctttattgatttttatttatacgTGTTTGAATAACTTGGCCAAATCGCCGAGAAG').upper()
    ]
lifespans = [
    0.038,
    0.032,
    0.041
    ]  # Example lifespans associated with the sequences

# Encoding the sequences
encoded_sequences = [encode_dna(seq, nucleotide_to_int, 1000) for seq in sample_sequences]
X_train = torch.tensor(encoded_sequences, dtype=torch.long)
y_train = torch.tensor(lifespans, dtype=torch.float32)

# Train the model
model = train_model(X_train, y_train)

# Predict lifespan for a new DNA sequence
new_dna = ('GAATAGAATACTGGACGACATTGtacatattttccaaaaaatcagaaagtaGATGACGGGACCAATTCTTTCTGTCAGGT' +
           'TTTACAACCGCCCAGTGCGTCTACGTCACATGTTGTATAAATGGTTGTAAACAATATGCGGAAACAATCAAATGCATTCC' +
           'CATAAGGCATAATATAGAGGCTACAGGCAATGAGTATCGCTCTTTGCTTTgtttaaagggggagtagagtttgtggggaa' +
           'atatatgtttctgactctaattttgcccctgataccgaatatcgatgtgaaaaaatttaaaaaaatttccctgattttat' +
           'attaatttttaaaatccgaaaatccattggatgcctatatgtgagtttttaaacgcaaaattttcccggcagagacgccc' +
           'cgcccacgaaaccgtgccgcacgtgtgggtttacgagctgaatattttccttctatttttatttgattttataccgattt' +
           'tcgtcgatttttctcattttttctcttttttttggtgttttttattgaaaattttgtgattttcgtaaatttattcctat' +
           'ttattaataaaaacaaaaacaattccatTAAATATCCCATTTTCAGCGCAAAATCGACTGGAGACTAGGAAAATCGTCTG').upper()
predicted_lifespan = predict_lifespan(model, new_dna, nucleotide_to_int)
print(f"Predicted Lifespan: {predicted_lifespan:.2f} years")

# Optimize lifespan by mutation
best_dna, best_lifespan = optimize_lifespan(model, new_dna, nucleotide_to_int)
print(f"Optimized DNA: {best_dna}")
print(f"Predicted Optimized Lifespan: {best_lifespan:.2f} years")