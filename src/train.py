import sys
import yaml
import torch
import pandas as pd
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ANN_pricer import ANN

config_path = PROJECT_ROOT / "configs" / "model_training.yaml"
model_config_path = PROJECT_ROOT / "configs" / "model_architecture.yaml"


#################### read the config and create some variables
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cfg_data = config["data"]
shuffle = cfg_data["shuffle"]
seed = config["meta"]["seed"]

data_path = PROJECT_ROOT / cfg_data["dir"]
data_df = pd.read_parquet(data_path)

#################### train the model
### read the data
df_X = data_df.iloc[:, :-1]
df_y = data_df.iloc[:, -1]

X = torch.from_numpy(df_X.values).float()
y = torch.from_numpy(df_y.values).float().view(-1, 1)

device = "mps" if torch.backends.mps.is_available() else "cpu"

### split train/val/test
dataset = TensorDataset(X, y)
N = len(dataset)
n_train = int(N * cfg_data["splits"]["train"])
n_val = int(N * cfg_data["splits"]["val"])
n_test = N - n_train - n_val        # more robust
train_ds, val_ds, test_ds = random_split(
    dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
)

batch_size = config["loop"]["batch_size"]
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)

### load the model
with open(model_config_path, "r") as f:
    model_cfg = yaml.safe_load(f)


model = ANN(
    input_dim=model_cfg["input"]["dim"],
    hidden_dims=model_cfg["hidden"]["dims"],
    output_dim=model_cfg["output"]["dim"],
    activation=model_cfg["hidden"]["activation"],
    dropout_rate=model_cfg["hidden"]["dropout_rate"],
    initialization=model_cfg["hidden"]["initialization"],
).to(device)

### loss 
loss_name = (config["loss"]["name"]).lower()
if loss_name == "mse":
    loss_fn = nn.MSELoss()
elif loss_name == "mae":
    loss_fn = nn.L1Loss()
elif loss_name == "nll":
    loss_fn = nn.NLLLoss()
elif loss_name == "cross_entropy":
    loss_fn = nn.CrossEntropyLoss()
else:
    raise ValueError(f"Loss function: '{loss}' not implemented")

### optimizer
#NOTE: implementar mejor si quiero usar otro optmizer
opt_name = (config["optimizer"]["name"]).lower()
if opt_name == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["learn_rate"],
        weight_decay=config["optimizer"]["weight_decay"]
        )
else: 
    raise ValueError(f"Optimizer: '{opt_name}' not implemented")

### training with validation
epochs = config["loop"]["epochs"]

for epoch in range(1, epochs+1): # each epoch
    # train
    model.train()
    train_sum = 0.0

    for xb, yb in train_loader: # each batch
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad() # para no acumular los gradientes antiguos
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        train_sum += loss.item() * xb.size(0) # loss.item() es una media del escalar de loss para una epoch

    train_loss = train_sum / n_train 


    # validation
    model.eval()
    val_sum = 0.0
    with torch.no_grad(): 
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_sum += loss_fn(pred, yb).item() * xb.size(0)

    val_loss = val_sum / n_val

    if epoch==1 or epoch%5==0:
        print(
            f"Epoch {epoch:3d}/{epochs} | train {loss_name}: {train_loss:.6f} | val {loss_name}: {val_loss:.6f}"
        )

### compute metrics

### callbacks

### prints/logs

#TODO: donde pongo los callbacks?
#################### save the outputs
# the trained model
# metrics of the vallidation
# the test_ds
