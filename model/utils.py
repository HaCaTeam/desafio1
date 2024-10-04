import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def train(model, dataloader, criterion, device, optimizer=None, has_grad=True):
    model.train()
    train_loss = 0
    for user_ids, item_ids, similarities in tqdm(dataloader, desc="Training"):
        user_ids, item_ids, similarities = user_ids.to(device), item_ids.to(device), similarities.to(device).float()
        if has_grad:
            optimizer.zero_grad()
        
        outputs = model(user_ids, item_ids)
        
        if has_grad:
            loss = criterion(outputs, similarities)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
    return train_loss / len(dataloader)

def predict(model, user_id, item_id, device):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id]).to(device)
        item_tensor = torch.tensor([item_id]).to(device)
        similarity = model(user_tensor, item_tensor)
        return similarity.item()