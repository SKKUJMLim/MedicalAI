import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MINE(nn.Module):
    def __init__(self, dim_z, dim_y=1, hidden_size=128):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(dim_z + dim_y, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, z, y):
        xy = torch.cat([z, y], dim=1)
        h = torch.relu(self.fc1(xy))
        return self.fc2(h)

def estimate_mutual_information(z, y, mine_net, optimizer, epochs=300, batch_size=128):
    dataset = TensorDataset(z, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for z_batch, y_batch in dataloader:
            joint = mine_net(z_batch, y_batch)
            y_shuffle = y_batch[torch.randperm(y_batch.size(0))]
            marginal = mine_net(z_batch, y_shuffle)

            loss = - (torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal))))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        joint = mine_net(z, y)
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal = mine_net(z, y_shuffle)
        mi = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
    return mi.item()

def extract_features_and_labels(model, dataloader, device):
    model.eval()
    z_img_list, z_total_list, y_list = [], [], []

    with torch.no_grad():
        for ids, preap_inputs, prelat_inputs, clinic_inputs, labels in dataloader:
            preap_inputs = preap_inputs.to(device)
            prelat_inputs = prelat_inputs.to(device)
            clinic_inputs = clinic_inputs.to(device)
            labels = labels.to(device)

            z_ap = model.model1(preap_inputs)
            z_lat = model.model2(prelat_inputs)

            z_img = torch.cat([z_ap, z_lat], dim=1).cpu()
            z_total = torch.cat([z_ap, z_lat, clinic_inputs], dim=1).cpu()
            y = labels.view(-1, 1).float().cpu()

            z_img_list.append(z_img)
            z_total_list.append(z_total)
            y_list.append(y)

    return (
        torch.cat(z_img_list, dim=0),
        torch.cat(z_total_list, dim=0),
        torch.cat(y_list, dim=0)
    )

def run_mine_analysis(model, dataloader, device):
    print("\n🔍 Estimating Mutual Information...")

    z_image_all, z_total_all, y_all = extract_features_and_labels(model, dataloader, device)

    mine_img = MINE(dim_z=z_image_all.shape[1]).to(device)
    opt_img = optim.Adam(mine_img.parameters(), lr=1e-3)
    mi_img = estimate_mutual_information(z_image_all.to(device), y_all.to(device), mine_img, opt_img)
    print(f"Mutual Information (Image-only): {mi_img:.4f}")

    mine_total = MINE(dim_z=z_total_all.shape[1]).to(device)
    opt_total = optim.Adam(mine_total.parameters(), lr=1e-3)
    mi_total = estimate_mutual_information(z_total_all.to(device), y_all.to(device), mine_total, opt_total)
    print(f"Mutual Information (Image + Clinical): {mi_total:.4f}")
