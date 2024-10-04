import json
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from tqdm import tqdm


TOPICS = ['Books and Literature', 'Environment', 'Careers', 'Healthy Living', 'Shopping', 'Automotive', 'Sensitive Topics', 'Politics', 'Events & Attractions', 'Maps & Navigation', 'Personal Celebrations & Life Events', 'Style & Fashion', 'Sports', 'Business and Finance', 'War and Conflicts', 'Hobbies & Interests', 'Education', 'Food & Drink', 'Real Estate', 'Crime, Law & Justice', 'Communication', 'Family and Relationships', 'Disasters', 'Home & Garden', 'Video Gaming', 'Pets', 'Medical Health', 'Entertainment', 'Fine Art', 'Religion & Spirituality', 'Travel']

personas_vectors = []
programs_vectors = []

personas_info = {}
programs_info = {}

with open('../data/personas/personas.json', 'rb') as f:
    personas = json.load(f)

    for p in personas:
        persona_vector = [0] * len(TOPICS)
        
        for topic in p["interests"]:
            persona_vector[TOPICS.index(topic)] = p["interests"][topic]
        
        personas_vectors.append({"name": p["name"], "vector": persona_vector})

        personas_info[p["name"]] = p["interests"]
    

with open('../data/programs/programs_info.json', 'rb') as f:
    programs = json.load(f)

    for p in programs:
        program_vector = [0] * len(TOPICS)
        
        for topic in p["Topics"]:
            program_vector[TOPICS.index(topic["description"])] = topic["percentage"]

        programs_vectors.append({"name": p["Title"], "vector": program_vector})

        programs_info[p["Title"]] = p

# compute the similarity between each persona and each program
similarity_matrix = []

for p in personas_vectors:
    row = {}

    for pr in programs_vectors:
        similarity = np.dot(p["vector"], pr["vector"]) / (np.linalg.norm(p["vector"]) * np.linalg.norm(pr["vector"])) # cosine similarity

        row[pr["name"]] = similarity

    similarity_matrix.append(row)


df = pd.DataFrame(similarity_matrix, index=[p["name"] for p in personas_vectors])

class ProgramDataset(Dataset):
    def __init__(self, similarities):
        self.similarities = similarities
        self.users = similarities.index.values
        self.items = similarities.columns.values
        self.ds_size = len(self.users) * len(self.items)

    def __len__(self):
        return len(self.similarities.values.flatten())

    def __getitem__(self, idx):
        user_row = idx // len(self.items)
        item_col = idx % len(self.items)
        cell = self.similarities.values[user_row, item_col]
        return (user_row, item_col, cell)

dataset = ProgramDataset(df)

batch_size = 1 # equivalent to stochastic gradient descent
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class NonLinearModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NonLinearModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, user_ids, item_ids):
        # user_embeds = self.user_embedding(user_ids) -> for embeddings of reduced dimension
        # item_embeds = self.item_embedding(item_ids) -> for embeddings of reduced dimension
        user_embeds = torch.tensor([personas_vectors[user_id]["vector"] for user_id in user_ids]).to(device)
        item_embeds = torch.tensor([programs_vectors[item_id]["vector"] for item_id in item_ids]).to(device)

        x = torch.cat([user_embeds, item_embeds], dim=1)
        x = torch.relu(self.fc1(x)) # Dense layer with ReLU activation
        x = torch.relu(self.fc2(x)) # Dense layer with ReLU activation
        x = self.fc3(x) # Dense layer with linear activation
        
        return x.squeeze()
    
num_users = len(personas_vectors)
num_items = len(programs_vectors)
embedding_dim = len(TOPICS) # reduce to create a latent space
learning_rate = 1e-3
num_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NonLinearModel(num_users, num_items, embedding_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()
metric = RMSELoss()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for user_ids, item_ids, similarities in tqdm(dataloader, desc="Training"):
        user_ids, item_ids, similarities = user_ids.to(device), item_ids.to(device), similarities.to(device).float()
        optimizer.zero_grad()
        
        outputs = model(user_ids, item_ids)
        loss = criterion(outputs, similarities)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    return train_loss / len(dataloader)


train_loss = train(model, loader, optimizer, criterion, device)
print(f"Train Loss: {train_loss:.4f}")

def predict(model, user_id, item_id, device):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id]).to(device)
        item_tensor = torch.tensor([item_id]).to(device)
        similarity = model(user_tensor, item_tensor)
        return similarity.item()

# predict for "David"
david_id = df.index.get_loc("David")

# (item_id, predicted_similarity)
predictions = [(program, predict(model, david_id, df.columns.get_loc(program), device)) for program in df.columns]
predictions.sort(key=lambda x: x[1], reverse=True)
print(predictions)

"""

def sort_programs_by_similarity(persona_name):
    return df.loc[persona_name].sort_values(ascending=False)

# export for "David"
programs = sort_programs_by_similarity("David").head(10)

programs_obj = [programs_info[program] for program in programs.index]

data_obj = {
    "preferences": personas_info["David"],
    "programs": programs_obj
}

with open('../data/david_example.json', 'w') as f:
    json.dump(data_obj, f, indent=4)
"""