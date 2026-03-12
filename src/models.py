import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size=768, hidden_layers=[256, 128, 64], num_wdl_classes=5, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_size = size
        self.backbone = nn.Sequential(*layers)
        
        self.wdl_head = nn.Linear(last_size, num_wdl_classes)
        self.dtz_head = nn.Linear(last_size, 1)

    def forward(self, x):
        features = self.backbone(x)
        wdl_logits = self.wdl_head(features)
        dtz_val = self.dtz_head(features)
        return wdl_logits, dtz_val

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=3, num_wdl_classes=5,
                 first_omega_0=30, hidden_omega_0=30, dropout=0.1):
        super(SIREN, self).__init__()
        self.net = []
        self.net.append(SineLayer(input_size, hidden_size, is_first=True, omega_0=first_omega_0))

        for i in range(num_layers - 1):
            self.net.append(SineLayer(hidden_size, hidden_size, is_first=False, omega_0=hidden_omega_0))
            # Add dropout after each hidden layer (except the last one)
            if dropout > 0 and i < num_layers - 2:
                self.net.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*self.net)
        
        self.wdl_head = nn.Linear(hidden_size, num_wdl_classes)
        self.dtz_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.net(x)
        wdl_logits = self.wdl_head(features)
        dtz_val = self.dtz_head(features)
        return wdl_logits, dtz_val


def get_model_for_endgame(model_type: str, num_pieces: int, num_wdl_classes: int = 5):
    """
    Returns an appropriate model for the given endgame configuration.
    
    Args:
        model_type: 'mlp' or 'siren'
        num_pieces: Number of pieces in the endgame (e.g., 3 for KQvK)
        num_wdl_classes: Number of WDL classes (default: 5)
    
    Returns:
        Model instance with appropriate architecture
    """
    # Compact encoding: num_pieces * 64 dimensions
    input_size = num_pieces * 64
    
    if model_type == "mlp":
        # Simpler architecture for simple endgames
        if num_pieces <= 3:
            # Very simple endgame: small model
            hidden_layers = [64, 32]
        elif num_pieces <= 4:
            # 4-piece endgame: medium model
            hidden_layers = [128, 64, 32]
        else:
            # 5+ piece endgame: larger model
            hidden_layers = [256, 128, 64]
        
        return MLP(input_size=input_size, hidden_layers=hidden_layers,
                   num_wdl_classes=num_wdl_classes, dropout=0.1)
    
    elif model_type == "siren":
        # SIREN architecture with dropout for regularization
        if num_pieces <= 3:
            hidden_size = 64
            num_layers = 2
            dropout = 0.0  # No dropout for very small models
        elif num_pieces <= 4:
            hidden_size = 128
            num_layers = 3
            dropout = 0.1
        else:
            hidden_size = 256
            num_layers = 4
            dropout = 0.1
        
        return SIREN(input_size=input_size, hidden_size=hidden_size,
                     num_layers=num_layers, num_wdl_classes=num_wdl_classes,
                     dropout=dropout)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
