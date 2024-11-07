import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import requests
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embedding_dim = 64  # Dimension of embeddings for each token
hidden_size = 32
num_layers = 1
seq_length = 15
dropout_prob = 0.2
batch_size = 128
learning_rate = 0.001
num_epochs = 40
patience = 5
seed = 42
hidden_size_gru = 128
hidden_size_lstm = 512
num_gru_layers = 2
num_lstm_layers = 2

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



class Transfered_HybridModel(nn.Module):
    def __init__(self, base_model, output_size):
        super(Transfered_HybridModel, self).__init__()

        # Load the base model
        self.embedding = base_model.embedding
        self.lstm = base_model.lstm
        self.gru = base_model.gru
        self.fc = base_model.fc

        # Freeze LSTM and Dropout layers
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.gru.parameters():
            param.requires_grad = False

        # Replace the final fully connected (dense) layer
        input_features = base_model.fc.in_features  # Get input size of original fully connected layer
        self.fc = nn.Linear(input_features, output_size)  # New fully connected layer with output_size = 34

    def forward(self, x):
        x = self.embedding(x)  # Convert input indices to embeddings
        
        # Initialize hidden states for GRU and LSTM
        h_gru = torch.zeros(2, x.size(0), 128).to(device)  # 2 layers of GRU with 128 units
        h_lstm = (torch.zeros(2, x.size(0), 512).to(device),  # 2 layers of LSTM with 512 units
                    torch.zeros(2, x.size(0), 512).to(device))
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h_gru)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(gru_out, h_lstm)
        
        # Take the last time step output and pass it through the fully connected layer
        out = self.fc(lstm_out[:, -1, :])
        return out
    
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

class HybridModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size_gru, hidden_size_lstm, num_gru_layers, num_lstm_layers, dropout_prob):
        super(HybridModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        
        # GRU layers
        self.gru = nn.GRU(embedding_dim, hidden_size_gru, num_gru_layers, batch_first=True, dropout=dropout_prob)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_size_gru, hidden_size_lstm, num_lstm_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layer to output logits
        self.fc = nn.Linear(hidden_size_lstm, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert input indices to embeddings
        
        # Initialize hidden states for GRU and LSTM
        h_gru = torch.zeros(2, x.size(0), 128).to(device)  # 2 layers of GRU with 128 units
        h_lstm = (torch.zeros(2, x.size(0), 512).to(device),  # 2 layers of LSTM with 512 units
                  torch.zeros(2, x.size(0), 512).to(device))
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h_gru)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(gru_out, h_lstm)
        
        # Take the last time step output and pass it through the fully connected layer
        out = self.fc(lstm_out[:, -1, :])
        return out

output_size = 34

DATA_PATH = "data/alzheimersdata.txt"
print("Loading data...")
data = read(DATA_PATH)  # Load SMILES data from file
# data = data[:100]  # For testing, limit the data size

# Tokenizer initialization and fitting
smiles_tokenizer = SMILESTokenizer()
smiles_tokenizer.fit(data)
vocab_size = smiles_tokenizer.vocab_size()
# vocab_size = 81

print(vocab_size)
# Split the data into train, test, and validation sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=seed)

# Create datasets and loaders
train_dataset = ChemicalDataset(train_data, smiles_tokenizer)
val_dataset = ChemicalDataset(val_data, smiles_tokenizer)
test_dataset = ChemicalDataset(test_data, smiles_tokenizer)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

base_model = HybridModel(vocab_size, embedding_dim, hidden_size_gru, hidden_size_lstm, num_gru_layers, num_lstm_layers, dropout_prob).to(device)
pytorch_total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print(pytorch_total_params)
print(base_model)
base_model.load_state_dict(torch.load('models/hybrid_model.pth'))

# Create the transfer learned model
transfer_model = Transfered_HybridModel(base_model,vocab_size)
print(transfer_model)
transfer_model.fc.requires_grad = True  # Allow training for final layer
transfer_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transfer_model.parameters(), lr=learning_rate)

# Early stopping variables
early_stopping_counter = 0
best_val_loss = float('inf')

# Training Loop
for epoch in range(num_epochs):
    transfer_model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = transfer_model(x_batch)
        loss = criterion(outputs, y_batch)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = epoch_loss / len(train_loader)

    # Validation
    transfer_model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            val_outputs = transfer_model(x_val_batch)
            val_loss += criterion(val_outputs, y_val_batch).item()
    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(transfer_model.state_dict(), 'models/transfered_hybrid_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break