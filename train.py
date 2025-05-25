import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformer import MusicTransformer
import yaml, os

class MIDIDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for fname in os.listdir(data_dir):
            with open(os.path.join(data_dir, fname)) as f:
                tokens = [int(tok) for tok in f.read().split()]
                self.samples.append(torch.tensor(tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx][:-1]
        y = self.samples[idx][1:]
        return x, y

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MIDIDataset(cfg["data_dir"])
dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

model = MusicTransformer(cfg["vocab_size"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

for epoch in range(cfg["epochs"]):
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), cfg["save_path"])


def train():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MIDIDataset(cfg["data_dir"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = MusicTransformer(cfg["vocab_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), cfg["save_path"])

if __name__ == "__main__":
    train()
