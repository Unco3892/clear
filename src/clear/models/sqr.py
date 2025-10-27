#!/usr/bin/env python3
"""
Improved SQR (Simultaneous Quantile Regression) for aleatoric uncertainty.
Key improvements:
1. Better network architecture with skip connections
2. Adaptive learning rate scheduling
3. Quantile crossing penalty to ensure proper ordering
4. Ensemble of SQR models for more robust estimates
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy


class QuantileNetwork(nn.Module):
    """
    Improved neural network for simultaneous quantile regression.
    Features skip connections and better regularization.
    """
    
    def __init__(self, input_dim, hidden_sizes=(256, 256, 128), dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        
        # Build layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_dim, hidden_size))
            prev_dim = hidden_size
        
        # Output heads for lower and upper quantiles
        self.lower_head = nn.Linear(prev_dim, 1)
        self.upper_head = nn.Linear(prev_dim, 1)
        
        # Skip connection from input to final layer
        self.skip = nn.Linear(input_dim, prev_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(h) for h in hidden_sizes])
        
    def forward(self, x):
        # Main path
        h = x
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norm)):
            h = layer(h)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Add skip connection
        skip = self.skip(x)
        h = h + skip
        
        # Output quantiles
        lower = self.lower_head(h)
        upper = self.upper_head(h)
        
        return lower, upper


class SQR:
    """
    Improved Simultaneous Quantile Regression for aleatoric uncertainty.
    Directly predicts the desired quantiles with enhanced training stability.
    """
    
    def __init__(self,
                 alpha=0.1,  # For 90% intervals: lower=0.05, upper=0.95
                 hidden_sizes=(256, 256, 128),
                 learning_rate=5e-4,
                 n_epochs=3000,
                 batch_size=128,
                 dropout=0.2,
                 weight_decay=1e-5,
                 crossing_penalty=1.0,
                 ensemble_size=1,
                 patience=200,
                  random_state=42,
                  verbose=False):
        self.alpha = alpha
        self.tau_lower = alpha / 2
        self.tau_upper = 1 - alpha / 2
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.crossing_penalty = crossing_penalty
        self.ensemble_size = ensemble_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        
    def _quantile_loss(self, pred, target, tau):
        """Pinball loss for quantile regression."""
        diff = target - pred
        loss = torch.where(diff >= 0, tau * diff, (tau - 1) * diff)
        return loss.mean()
    
    def _crossing_penalty_loss(self, lower, upper):
        """Penalty for quantile crossing (when lower > upper)."""
        # ReLU of the difference penalizes crossing
        penalty = F.relu(lower - upper + 0.01)  # Small margin to ensure separation
        return penalty.mean()
    
    def _combined_loss(self, lower, upper, y):
        """Combined loss with quantile losses and crossing penalty."""
        # Quantile losses
        loss_lower = self._quantile_loss(lower, y, self.tau_lower)
        loss_upper = self._quantile_loss(upper, y, self.tau_upper)
        
        # Crossing penalty
        crossing_loss = self._crossing_penalty_loss(lower, upper)
        
        # Total loss
        total_loss = loss_lower + loss_upper + self.crossing_penalty * crossing_loss
        
        return total_loss, {
            'lower': loss_lower.item(),
            'upper': loss_upper.item(),
            'crossing': crossing_loss.item()
        }
    
    def _train_single_model(self, X_train, y_train, X_val, y_val, seed):
        """Train a single SQR model."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standardize data
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_val_scaled = x_scaler.transform(X_val)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)
        
        # Create data loader with shuffling
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # Create model
        model = QuantileNetwork(
            X_train.shape[1], 
            self.hidden_sizes, 
            self.dropout
        ).to(device)
        
        # Initialize weights carefully
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        
        # Optimizer with AdamW for better regularization
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler - reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=False
        )
        
        # Training with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training
            model.train()
            epoch_losses = {'lower': 0, 'upper': 0, 'crossing': 0}
            
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                
                lower, upper = model(X_batch)
                loss, loss_components = self._combined_loss(
                    lower.squeeze(), upper.squeeze(), y_batch
                )
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                for k, v in loss_components.items():
                    epoch_losses[k] += v
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_lower, val_upper = model(X_val_t)
                val_loss, _ = self._combined_loss(
                    val_lower.squeeze(), val_upper.squeeze(), y_val_t
                )
                val_loss_value = val_loss.item()
            
            # Learning rate scheduling
            scheduler.step(val_loss_value)
            
            # Early stopping check
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience and epoch > 500:
                    if self.verbose:
                        print(f"    Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if self.verbose and (epoch % 200 == 0 or epoch == self.n_epochs - 1):
                n_batches = len(loader)
                print(f"    Epoch {epoch}: "
                      f"train_lower={epoch_losses['lower']/n_batches:.4f}, "
                      f"train_upper={epoch_losses['upper']/n_batches:.4f}, "
                      f"crossing={epoch_losses['crossing']/n_batches:.4f}, "
                      f"val_loss={val_loss_value:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        return model, x_scaler, y_scaler
    
    def fit(self, X, y):
        """Fit the SQR model(s)."""
        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        print(f"Training SQR with {self.ensemble_size} model(s)...")
        
        self.models = []
        self.scalers = []
        
        for i in range(self.ensemble_size):
            print(f"  Training SQR model {i+1}/{self.ensemble_size}...")
            model, x_scaler, y_scaler = self._train_single_model(
                X_train, y_train, X_val, y_val, 
                seed=self.random_state + i * 1000
            )
            self.models.append(model)
            self.scalers.append((x_scaler, y_scaler))
        
        # Validate coverage on validation set
        _, lower, upper = self.predict(X_val)
        coverage = np.mean((y_val >= lower) & (y_val <= upper))
        print(f"  Validation coverage: {coverage:.3f} (target: {1-self.alpha:.3f})")
        
        return self
    
    def predict(self, X):
        """Predict quantiles."""
        device = next(self.models[0].parameters()).device
        
        all_lower = []
        all_upper = []
        
        for model, (x_scaler, y_scaler) in zip(self.models, self.scalers):
            # Scale input
            X_scaled = x_scaler.transform(X)
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            
            # Predict
            with torch.no_grad():
                lower_scaled, upper_scaled = model(X_t)
                lower_scaled = lower_scaled.squeeze().cpu().numpy()
                upper_scaled = upper_scaled.squeeze().cpu().numpy()
            
            # Inverse transform
            lower = y_scaler.inverse_transform(lower_scaled.reshape(-1, 1)).flatten()
            upper = y_scaler.inverse_transform(upper_scaled.reshape(-1, 1)).flatten()
            
            all_lower.append(lower)
            all_upper.append(upper)
        
        # Average predictions if ensemble
        if self.ensemble_size > 1:
            lower = np.mean(all_lower, axis=0)
            upper = np.mean(all_upper, axis=0)
        else:
            lower = all_lower[0]
            upper = all_upper[0]
        
        # Ensure proper ordering
        median = (lower + upper) / 2
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)
        
        return median, lower, upper
