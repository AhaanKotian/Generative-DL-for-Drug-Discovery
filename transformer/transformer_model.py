#Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import requests
import os


def max_sequence(data):
	file = open(data)
	max_seq_len = int(len(max(file,key=len)))
	print ("Max Sequence Length: ", max_seq_len)
	return max_seq_len


# reads the lines in a file
def read(fileName):
        fileObj = open(fileName, "r")
        words = fileObj.read().splitlines()
        fileObj.close()
        return words


# Join inputed array into a single string
def concatenate(data):
    res = ''.join(data)
    return res


"""Baseline Model Generic Dataset training"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embedding_dim = 256  # Dimension of embeddings for each token
hidden_size = 2048
num_layers = 12
seq_length = 15
num_heads = 16
batch_size = 128
learning_rate = 0.001
num_epochs = 40
patience = 5
seed = 42

# SMILES Tokenizer
class SMILESTokenizer:
    def __init__(self):
        # Special characters in SMILES notation
        self.special_tokens = ['[', ']', '=', '(', ')', '@', '+', '-', '#', '%']
        self.atom_tokens = set('CNOFPSIHBrcnl123456789')
        self.token_to_idx = {}
        self.idx_to_token = {}

    def fit(self, smiles_list):
        '''
        Here we are iterating over all the smiles strings in smiles_list and tokenizing every smiles string in the list.
        Also we are creating the vocab of all the tokens that we are encountering while tokenizing the smiles string.
        Then create mappings from token to index and vice versa
        '''
        vocab = set()
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            vocab.update(tokens)

        # Create mappings from token to index and vice versa
        self.token_to_idx = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def tokenize(self, smiles):
        '''
        Explaining the tokenizer via an example
        CN1CCC[C@H]1c2cccnc2
        Given above is 1 smiles strings
        We are going to tokenize this string
        Tokens of the string are:
        ['C','N','1','C','C','C','[C@H]','1','c','2','c','c','c','n','c','2']
        '''
        tokens = []
        i = 0
        while i < len(smiles):
            if smiles[i] == '[':  #Here we check if there is an opening bracket in the string and we will append all the elements until we find a closing square bracket. Thus all the characters between 2 brackets will be considered as 1 token. For eg C@H is 1 token
                j = i + 1
                while j < len(smiles) and smiles[j] != ']':
                    j += 1
                tokens.append(smiles[i:j + 1])
                i = j + 1
            elif smiles[i:i+2] in ['Cl', 'Br']:  #We check if there are Elements with 2 char like Cl and Br
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                tokens.append(smiles[i]) #If both the cases above are not true then we will consider every character as 1 token
                i += 1
        return tokens

    # def encode(self, smiles):
    #     tokens = self.tokenize(smiles)
    #     return [self.token_to_idx[token] for token in tokens]
    def encode(self, smiles):
      tokens = self.tokenize(smiles)
      encoded = []
      for token in tokens:
          if token in self.token_to_idx:
              encoded.append(self.token_to_idx[token])
          else:
              continue

      return encoded

    def decode(self, indices):
        return ''.join([self.idx_to_token[idx] for idx in indices])

    def vocab_size(self):
        return len(self.token_to_idx)

# Custom Dataset class
class ChemicalDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.x_data, self.y_data = self.prepare_data()

    def prepare_data(self):
        x_data = []
        y_data = []
        for smile in self.data:
            encoded = self.tokenizer.encode(smile)
            for i in range(0, len(encoded) - seq_length):
                x_data.append(encoded[i:i+seq_length])
                y_data.append(encoded[i+seq_length])
        return np.array(x_data), np.array(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return torch.tensor(self.x_data[idx], dtype=torch.long), torch.tensor(self.y_data[idx], dtype=torch.long)
    
class LargeTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_length):
        super(LargeTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._generate_positional_encoding(max_seq_length, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:x.size(1), :].to(x.device)
        transformer_out = self.transformer_encoder(embedded)
        out = self.fc(transformer_out[:, -1, :])
        return out

    def _generate_positional_encoding(self, max_seq_length, embedding_dim):
        pos = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        positional_encoding = np.zeros((max_seq_length, embedding_dim))
        positional_encoding[:, 0::2] = np.sin(pos * div_term)
        positional_encoding[:, 1::2] = np.cos(pos * div_term)
        return torch.tensor(positional_encoding, dtype=torch.float32)



# Load and preprocess data
DATA_PATH = "/home/somaiya-1/Desktop/Generative-DL-for-Drug-Discovery/data/data.txt"
NEW_DATA = "/home/somaiya-1/Desktop/Generative-DL-for-Drug-Discovery/data/processed_data.txt"
TRAIN_DATA = '/home/somaiya-1/Desktop/Generative-DL-for-Drug-Discovery/data/train_data.txt'
TEST_DATA = '/home/somaiya-1/Desktop/Generative-DL-for-Drug-Discovery/data/test_data.txt'
VAL_DATA = '/home/somaiya-1/Desktop/Generative-DL-for-Drug-Discovery/data/val_data.txt'

# Filter Dataset Into Smaller Segments
count=0
new = open(NEW_DATA, "w")
for line in open(DATA_PATH, "r"):
    count += 1
    if 30 < len(line) < 50 and count < 300000:
        # Remove uncommon letters
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))

print("Loading data...")
data = read(NEW_DATA)  # Load SMILES data from file

# Tokenizer initialization and fitting
smiles_tokenizer = SMILESTokenizer()
smiles_tokenizer.fit(data)
vocab_size = smiles_tokenizer.vocab_size()

# Split the data into train, test, and validation sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=seed)

# Create datasets and loaders
train_dataset = ChemicalDataset(data, smiles_tokenizer)
val_dataset = ChemicalDataset(val_data, smiles_tokenizer)
test_dataset = ChemicalDataset(test_data, smiles_tokenizer)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = LargeTransformerModel(vocab_size, embedding_dim, num_heads, hidden_size, num_layers, max_seq_length=15).to(device)
# model = TransformerModel(vocab_size, embedding_dim, num_heads, hidden_size, num_layers, max_seq_length=15).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping variables
early_stopping_counter = 0
best_val_loss = float('inf')

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)

# Training Loop with Accuracy
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        epoch_loss += loss.item()

        # Accuracy computation
        accuracy = calculate_accuracy(outputs, y_batch)
        epoch_accuracy += accuracy

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_accuracy = epoch_accuracy / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)

            val_outputs = model(x_val_batch)
            val_loss += criterion(val_outputs, y_val_batch).item()

            # Validation accuracy computation
            val_accuracy += calculate_accuracy(val_outputs, y_val_batch)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), '/home/somaiya-1/Desktop/Generative-DL-for-Drug-Discovery/models/largetransformer.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break


# Training Loop
# for epoch in range(num_epochs):
#     model.train()
#     print(epoch)
#     epoch_loss = 0
#     for x_batch, y_batch in train_loader:
        
#         x_batch = x_batch.to(device)
#         y_batch = y_batch.to(device)

#         outputs = model(x_batch)
#         loss = criterion(outputs, y_batch)
#         epoch_loss += loss.item()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     avg_train_loss = epoch_loss / len(train_loader)

#     # Validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for x_val_batch, y_val_batch in val_loader:
#             x_val_batch = x_val_batch.to(device)
#             y_val_batch = y_val_batch.to(device)
#             val_outputs = model(x_val_batch)
#             val_loss += criterion(val_outputs, y_val_batch).item()
#     avg_val_loss = val_loss / len(val_loader)

#     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

#     # Early stopping check
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         early_stopping_counter = 0
#         torch.save(model.state_dict(), 'models/improvedlstm.pth')
#     else:
#         early_stopping_counter += 1
#         if early_stopping_counter >= patience:
#             print("Early stopping triggered")
#             break