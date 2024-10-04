from torch.utils.data import Dataset

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
