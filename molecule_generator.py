import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from tl_baseline import SMILESTokenizer, Transfered_BaselineModel, LSTMBaseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Function to read lines from a file
def read(fileName):
    with open(fileName, "r") as fileObj:
        words = fileObj.read().splitlines()
    return words

def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    model.eval()  # Set the model to evaluation mode
    generated_text = seed_text

    # Generate characters one at a time
    for _ in range(next_words):
        token_list = tokenizer.encode(generated_text)
        # print("Tokens:", token_list)  # Debugging line

        # Skip if there's an unknown token
        if -1 in token_list:
            # print("Unknown token encountered. Stopping generation.")
            # break
            continue
        
        token_list = torch.tensor(token_list[-max_sequence_len:]).unsqueeze(0).to(device)  # Ensure it fits the model input shape

        with torch.no_grad():
            # Get the model's predictions
            predictions = model(token_list)
            predicted_index = torch.argmax(predictions, dim=1).item()  # Get the predicted index
            predicted_char = tokenizer.idx_to_token[predicted_index]  # Map back to character

        generated_text += predicted_char

        # Stop if we encounter a special character (can be adjusted)
        if predicted_char == '!':
            break

    return generated_text

def checkSMILES(smiles):
    valid_smiles = []
    for smile in smiles:
        if Chem.MolFromSmiles(smile, sanitize=False) is not None and len(smile) > 15:
            valid_smiles.append(smile)
    return valid_smiles


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


# Load the trained model

# Generate molecules
ALZHEIMERS_DATA = 'data/alzheimersdata.txt'
ad_data = np.array(read(ALZHEIMERS_DATA))


smiles_tokenizer = SMILESTokenizer()
smiles_tokenizer.fit(ad_data)
vocab_size = smiles_tokenizer.vocab_size()
smiles = []

base_model = LSTMBaseline(81, embedding_dim, hidden_size, num_layers, dropout_prob).to(device)
print(base_model)
base_model.load_state_dict(torch.load('models/baselinelstm.pth'))



transfer_model = Transfered_BaselineModel(base_model,vocab_size)
print(transfer_model)
transfer_model.load_state_dict(torch.load('models/transfered_baseline_lstm.pth'))
transfer_model.to(device)
transfer_model.eval()


# Create 100 molecules
for i in range(100):
    if i % 10 == 0:
        print(f"Generating molecule {i+1}/100")
    start = np.random.randint(0, len(ad_data) - 1)
    single_seed = ad_data[start][:15]  # Use the first 15 characters as seed
    prediction = generate_text(single_seed, 35, transfer_model, smiles_tokenizer, seq_length)
    print("Generated SMILES:", prediction)
    smiles.append(prediction)
print("Done")
# print("Unknown tokens encountered during encoding:", smiles_tokenizer.unknown_tokens)

# Filter valid molecules
generated_molecules = checkSMILES(smiles)
print("Valid Generated Molecules:", generated_molecules)