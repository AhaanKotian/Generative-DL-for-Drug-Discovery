# import numpy as np
# import torch
# import torch.nn.functional as F
# from rdkit import Chem
# # from improved.tl_improved import SMILESTokenizer, Transfered_ImprovedLSTM, LSTMComplex
# from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import requests
import os
from rdkit import Chem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Function to read lines from a file
def read(fileName):
    with open(fileName, "r") as fileObj:
        words = fileObj.read().splitlines()
    return words

def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    model.eval()  # Set the model to evaluation mode
    generated_text = seed_text

    for _ in range(next_words):
        token_list = tokenizer.encode(generated_text)
        
        # Skip if there's an unknown token
        if -1 in token_list:
            print("Unknown token encountered. Skipping iteration.")
            continue
        
        token_list = torch.tensor(token_list[-max_sequence_len:]).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(token_list)
            predicted_index = torch.argmax(predictions, dim=1).item()
            predicted_char = tokenizer.idx_to_token[predicted_index] 
            # # Ensure no KeyError occurs
            # predicted_char = tokenizer.idx_to_token.get(predicted_index, '<UNK>')  # Fallback to '<UNK>' if index not found

        generated_text += predicted_char

        if predicted_char == '!':  # Optional stopping condition
            break

    return generated_text


def checkSMILES(smiles):
    valid_smiles = []
    for smile in smiles:
        if Chem.MolFromSmiles(smile, sanitize=False) is not None and len(smile) > 15:
            valid_smiles.append(smile)
    return valid_smiles



#baseline
# embedding_dim = 64  # Dimension of embeddings for each token
# hidden_size = 32
# num_layers = 1
# seq_length = 15
# dropout_prob = 0.2
# batch_size = 128
# learning_rate = 0.001
# num_epochs = 40
# patience = 5
# seed = 42

#hybrid
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
num_gru_layers = 3
num_lstm_layers = 3

#improved
# embedding_dim = 64  # Dimension of embeddings for each token
# hidden_size = 512
# num_layers = 5
# seq_length = 15
# dropout_prob = 0.2
# batch_size = 128
# learning_rate = 0.001
# num_epochs = 40
# patience = 5
# seed = 42

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
        h_gru = torch.zeros(3, x.size(0), 128).to(device)  # 2 layers of GRU with 128 units
        h_lstm = (torch.zeros(3, x.size(0), 512).to(device),  # 2 layers of LSTM with 512 units
                    torch.zeros(3, x.size(0), 512).to(device))
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h_gru)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(gru_out, h_lstm)
        
        # Take the last time step output and pass it through the fully connected layer
        out = self.fc(lstm_out[:, -1, :])
        return out


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
        h_gru = torch.zeros(3, x.size(0), 128).to(device)  # 2 layers of GRU with 128 units
        h_lstm = (torch.zeros(3, x.size(0), 512).to(device),  # 2 layers of LSTM with 512 units
                  torch.zeros(3, x.size(0), 512).to(device))
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h_gru)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(gru_out, h_lstm)
        
        # Take the last time step output and pass it through the fully connected layer
        out = self.fc(lstm_out[:, -1, :])
        return out
    

class LSTMComplex(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(LSTMComplex, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert input indices to embeddings
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Hidden state
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Cell state
        out, _ = self.lstm(x, (h_0, c_0))  # LSTM output
        out = out[:, -1, :]  # Take the last time step output
        out = self.fc(out)  # Fully connected layer to output logits
        return out

class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout_prob):
        super(LSTMBaseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert input indices to embeddings
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Hidden state
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Cell state
        out, _ = self.lstm(x, (h_0, c_0))  # LSTM output
        out = self.dropout(out[:, -1, :])  # Take the last time step output
        out = self.fc(out)  # Fully connected layer to output logits
        return out
# Load the trained model

# Generate molecules
#ALZHEIMERS_DATA = 'data/alzheimersdata.txt'
ALZHEIMERS_DATA = 'data/processed_data.txt'
ad_data = np.array(read(ALZHEIMERS_DATA))


smiles_tokenizer = SMILESTokenizer()
smiles_tokenizer.fit(ad_data)
vocab_size = smiles_tokenizer.vocab_size()
smiles = []
vocab_size = 81


# base_model = LSTMBaseline(vocab_size, embedding_dim, hidden_size, num_layers,dropout_prob).to(device)
base_model = HybridModel(vocab_size, embedding_dim, hidden_size_gru, hidden_size_lstm, num_gru_layers, num_lstm_layers, dropout_prob).to(device)
# base_model = LSTMComplex(vocab_size, embedding_dim, hidden_size, num_layers).to(device)
print(base_model)
base_model.load_state_dict(torch.load('models/hybridmodel.pth'))
# base_model.load_state_dict(torch.load('models/improved_lstm_model.pth'))

base_model.to(device)
base_model.eval()



# transfer_model = Transfered_ImprovedLSTM(base_model,vocab_size)
# print(transfer_model)
# transfer_model.load_state_dict(torch.load('models/improved_lstm_model.pth'))
# transfer_model.to(device)
# transfer_model.eval()


# Create 100 molecules
for i in range(100):
    if i % 10 == 0:
        print(f"Generating molecule {i+1}/100")
    start = np.random.randint(0, len(ad_data) - 1)
    single_seed = ad_data[start][:15]  # Use the first 15 characters as seed
    prediction = generate_text(single_seed, 35, base_model, smiles_tokenizer, seq_length)
    print("Generated SMILES:", prediction)
    smiles.append(prediction)
print("Done")
# print("Unknown tokens encountered during encoding:", smiles_tokenizer.unknown_tokens)

# Filter valid molecules
generated_molecules = checkSMILES(smiles)
print("Valid Generated Molecules:", generated_molecules)