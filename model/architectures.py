import torch
import torch.nn as nn

class NonLinearModel(nn.Module):
    def __init__(self, device, num_users, num_items, embedding_dim, personas_vectors, programs_vectors, latent_space=False):
        super(NonLinearModel, self).__init__()
        self.device = device
        self.latent_space = latent_space

        if self.latent_space:
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
        else:
            self.personas_vectors = personas_vectors
            self.programs_vectors = programs_vectors

        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, user_ids, item_ids):
        if self.latent_space:
            user_embeds = self.user_embedding(user_ids)
            item_embeds = self.item_embedding(item_ids)
        else:
            user_embeds = torch.tensor([self.personas_vectors[user_id]["vector"] for user_id in user_ids]).to(self.device)
            item_embeds = torch.tensor([self.programs_vectors[item_id]["vector"] for item_id in item_ids]).to(self.device)

        x = torch.cat([user_embeds, item_embeds], dim=1)
        x = torch.relu(self.fc1(x)) # Dense layer with ReLU activation
        x = torch.relu(self.fc2(x)) # Dense layer with ReLU activation
        x = self.fc3(x) # Dense layer with linear activation

        return x.squeeze()
    
class CossineSimilarityBlock(nn.Module):
    def __init__(self, device, num_users, num_items, embedding_dim, personas_vectors, programs_vectors):
        super(CossineSimilarityBlock, self).__init__()
        self.device = device
        self.personas_vectors = personas_vectors
        self.programs_vectors = programs_vectors
    
    def forward(self, user_ids, item_ids):
        user_embeds = torch.tensor([self.personas_vectors[user_id]["vector"] for user_id in user_ids]).to(self.device)
        item_embeds = torch.tensor([self.programs_vectors[item_id]["vector"] for item_id in item_ids]).to(self.device)
        x = torch.nn.functional.cosine_similarity(user_embeds, item_embeds, dim=1)

        return x.squeeze()
    
class MetaModel(nn.Module):
    def __init__(self, device, models):
        super(MetaModel, self).__init__()
        self.models = models
        self.device = device
        
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, user_ids, item_ids):
        x = torch.Tensor([model(user_ids, item_ids) for model in self.models]).to(self.device)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze()
