
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Activation: '{name}' not implemented")


        
class ANN(nn.Module):
    """
    MLP like Liu et al.
        8 -> 200 -> 200 -> 200 -> 200 -> 1
        Activation: ReLu on hidden layers
        Output: linear (regression)
    """

    def __init__(
            self, 
            input_dim: int = 8,
            hidden_dims=(200, 200, 200, 200),
            output_dim: int = 1,
            activation= "relu",
            dropout_rate: float = 0.0,
            initialization: str = "xavier_uniform"
            ):
        super().__init__()

        h1, h2, h3, h4 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, output_dim)

        self.act = get_activation(activation)
        self.drop = nn.Dropout(p=dropout_rate)

        self._init_weights(initialization)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.drop(self.act(self.fc3(x)))
        x = self.drop(self.act(self.fc4(x)))
        x = self.fc5(x) # linear ouput for regression
        return x

    def _init_weights(self, initialization: str) -> None:
        init = initialization.lower()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif init == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif  init == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight)
                elif init == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise ValueError(f"Initialization:'{initialization}' not supported")
                
                nn.init.zeros_(m.bias)