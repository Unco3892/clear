#!/usr/bin/env python3
"""
Pure Epistemic Deep Ensemble with enhanced diversity and calibrated intervals.
Key improvements:
1. Multiple diversity techniques (different seeds, architectures, dropout, data subsampling)
2. Calibrated confidence multiplier instead of fixed z-scores
3. Optional bootstrap aggregation for additional diversity
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy


class PureEpistemicEnsemble:
    """
    Pure epistemic uncertainty via deep ensemble of point-prediction networks.
    No aleatoric uncertainty - only captures model uncertainty through ensemble diversity.
    """
    
    def __init__(self, 
                 n_members=10,
                 hidden_sizes=(256, 128),
                 learning_rate=1e-3,
                 n_epochs=1000,
                 batch_size=64,
                 dropout_rate=0.1,
                 weight_decay=1e-5,
                 diversity_strategies=None,
                 calibrate_c=True,
                 random_state=42,
                 verbose=False):
        self.n_members = n_members
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.calibrate_c = calibrate_c
        self.random_state = random_state
        self.verbose = verbose
        
        # Default diversity strategies
        if diversity_strategies is None:
            diversity_strategies = [
                'random_init',      # Different random initializations
                'architecture',     # Vary architecture slightly
                'dropout',          # Different dropout rates
                'bootstrap',        # Bootstrap sampling
                'lr_schedule',      # Different learning rate schedules
                'data_augment'      # Small input noise
            ]
        self.diversity_strategies = diversity_strategies
        
        # Will be set during calibration
        self.c_multiplier = 2.0  # Default to ~95% coverage under Gaussian assumption
        
    def _create_network(self, input_dim, member_id):
        """Create a point-prediction network with diversity based on member ID."""
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers with diversity
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Architecture diversity: vary layer width
            if 'architecture' in self.diversity_strategies:
                size_variation = int(hidden_size * 0.2 * ((member_id % 3) - 1) / 1.5)
                hidden_size = max(32, hidden_size + size_variation)
            
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            
            # Dropout diversity: vary dropout rate
            if 'dropout' in self.diversity_strategies:
                dropout_variation = self.dropout_rate * (0.5 + (member_id % 5) * 0.25)
                layers.append(nn.Dropout(min(0.5, dropout_variation)))
            else:
                layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _get_learning_rate(self, member_id):
        """Learning rate diversity."""
        if 'lr_schedule' in self.diversity_strategies:
            # Vary learning rate by up to 50%
            lr_multiplier = 0.75 + (member_id % 4) * 0.25
            return self.learning_rate * lr_multiplier
        return self.learning_rate
    
    def _train_member(self, X_train, y_train, X_val, y_val, member_id, device):
        """Train a single ensemble member with enhanced diversity."""
        # Set unique random seed for this member
        torch.manual_seed(self.random_state + member_id * 1000)
        np.random.seed(self.random_state + member_id * 1000)
        
        # Bootstrap sampling for diversity
        if 'bootstrap' in self.diversity_strategies:
            n_samples = len(X_train)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        # Standardize data (each member gets its own scaler for diversity)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_val_scaled = x_scaler.transform(X_val)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Add small input noise for diversity
        if 'data_augment' in self.diversity_strategies:
            noise_scale = 0.01 * np.std(X_train_scaled, axis=0)
            X_train_scaled = X_train_scaled + np.random.randn(*X_train_scaled.shape) * noise_scale
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)
        
        # Create data loader
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create network
        model = self._create_network(X_train.shape[1], member_id).to(device)
        
        # Different initialization for diversity
        if 'random_init' in self.diversity_strategies:
            # Use different initialization schemes
            init_methods = [nn.init.xavier_normal_, nn.init.kaiming_normal_, 
                           nn.init.xavier_uniform_, nn.init.kaiming_uniform_]
            init_method = init_methods[member_id % len(init_methods)]
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    init_method(m.weight)
        
        # Optimizer with member-specific learning rate
        lr = self._get_learning_rate(member_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        
        # Learning rate scheduler for diversity
        if 'lr_schedule' in self.diversity_strategies:
            # Different schedulers for different members
            if member_id % 3 == 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
            elif member_id % 3 == 1:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.n_epochs//3, gamma=0.5)
            else:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        else:
            scheduler = None
        
        # Training with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience = 50
        patience_counter = 0
        
        model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch).squeeze()
                loss = nn.MSELoss()(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            if scheduler is not None:
                scheduler.step()
            
            # Validation for early stopping
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t).squeeze()
                    val_loss = nn.MSELoss()(val_pred, y_val_t).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience and epoch > 100:
                        if self.verbose:
                            print(f"    Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                        break
                
                # Verbose output every 100 epochs
                if self.verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1):
                    avg_train_loss = epoch_loss / len(loader)
                    print(f"    Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, best_val={best_val_loss:.6f}")
                
                model.train()
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        return model, x_scaler, y_scaler
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit the ensemble and calibrate the confidence multiplier."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Split validation set if not provided
        if X_val is None:
            val_size = int(0.2 * len(X_train))
            indices = np.random.permutation(len(X_train))
            X_val = X_train[indices[:val_size]]
            y_val = y_train[indices[:val_size]]
            X_train = X_train[indices[val_size:]]
            y_train = y_train[indices[val_size:]]
        
        # Train ensemble members
        print(f"Training pure epistemic ensemble with {self.n_members} members...")
        self.models = []
        self.scalers = []
        
        for i in range(self.n_members):
            print(f"  Training member {i+1}/{self.n_members}...")
            model, x_scaler, y_scaler = self._train_member(
                X_train, y_train, X_val, y_val, i, device
            )
            self.models.append(model)
            self.scalers.append((x_scaler, y_scaler))
        
        # Calibrate confidence multiplier if requested
        if self.calibrate_c:
            self._calibrate_multiplier(X_val, y_val)
        
        return self
    
    def _calibrate_multiplier(self, X_val, y_val, target_coverage=0.95):
        """Calibrate the confidence multiplier c to achieve target coverage."""
        # Get ensemble predictions on validation set
        predictions = self._get_ensemble_predictions(X_val)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0, ddof=1)
        
        # Try different multipliers and find the one that gives target coverage
        # c_values = np.linspace(0.5, 4.0, 100)
        c_values = np.logspace(-1, 2, num=4001)
        coverages = []
        
        for c in c_values:
            lower = mean_pred - c * std_pred
            upper = mean_pred + c * std_pred
            coverage = np.mean((y_val >= lower) & (y_val <= upper))
            coverages.append(coverage)
        
        # Find c that gives closest coverage to target
        best_idx = np.argmin(np.abs(np.array(coverages) - target_coverage))
        self.c_multiplier = c_values[best_idx]
        actual_coverage = coverages[best_idx]
        
        print(f"  Calibrated c={self.c_multiplier:.3f} for {actual_coverage:.3f} coverage (target: {target_coverage})")
    
    def _get_ensemble_predictions(self, X):
        """Get predictions from all ensemble members."""
        device = next(self.models[0].parameters()).device
        predictions = []
        
        for model, (x_scaler, y_scaler) in zip(self.models, self.scalers):
            X_scaled = x_scaler.transform(X)
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                pred_scaled = model(X_t).squeeze().cpu().numpy()
            
            # Inverse transform
            pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            predictions.append(pred)
        
        return np.array(predictions)  # Shape: (n_members, n_samples)
    
    def predict_interval(self, X, coverage=0.95):
        """Predict intervals using pure epistemic uncertainty."""
        predictions = self._get_ensemble_predictions(X)
        
        # Compute mean and std across ensemble
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0, ddof=1)
        
        # Use calibrated multiplier
        lower = mean_pred - self.c_multiplier * std_pred
        upper = mean_pred + self.c_multiplier * std_pred
        
        return mean_pred, lower, upper
    
    def predict(self, X):
        """Point prediction (ensemble mean)."""
        predictions = self._get_ensemble_predictions(X)
        return np.mean(predictions, axis=0)
