"""
Supporting models for the CLEAR framework which have not been tested and are not used in the paper. We aim to extend the framework to support these models in the future. We aim to provide a wrapper for scikit-learn like quantile regressors (future extension).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.model_selection import train_test_split
from pygam import ExpectileGAM, s, GAM
from tqdm import tqdm
# Import and configure tqdm for better progress bars
try:
    # Enable tqdm for non-notebook environments
    from tqdm.auto import tqdm as auto_tqdm  # This will use appropriate version based on environment
except ImportError:
    # Define dummy tqdm as fallback
    def tqdm(iterator, **kwargs):
        return iterator
    auto_tqdm = tqdm

# Rename RStyleQuantileGAM to QuantileGAM
class QuantileGAM:
    """
    Implementation of a quantile GAM using expectiles.
    """
    
    def __init__(self, quantile=0.5, n_splines=10, lam=None, random_state=None):
        """
        Initialize the quantile GAM model.
        
        Args:
            quantile: The quantile to estimate (between 0 and 1)
            n_splines: Number of splines to use
            lam: Smoothing parameter
            random_state: Random seed for reproducibility
        """
        self.quantile = quantile
        self.n_splines = n_splines
        self.lam = lam
        self.random_state = random_state
        self.model = ExpectileGAM(expectile=quantile, n_splines=n_splines, lam=lam)
        # print(f"Model parameters: {self.model.get_params()}")
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to the data.
        
        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Optional sample weights
        """
        # Convert to numpy array if needed
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Handle categorical variables
        # For each column with ≤9 unique values, convert to categorical
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        
        self.model.fit(X, y)
        self.is_fitted_ = True
        self.n_features_in_ = n_features
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted quantile values
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model not fitted yet. Call fit first.")
        
        X = np.asarray(X)
        return self.model.predict(X)

class QuantileLoss(torch.nn.Module):
    """PyTorch module implementing the quantile loss function."""
    def __init__(self):
        super(QuantileLoss, self).__init__()

    def forward(self, yhat, y, tau):
        """
        Compute the quantile loss.
        
        Args:
            yhat: The predicted values
            y: The target values
            tau: The quantile level (between 0 and 1)
            
        Returns:
            The quantile loss
        """
        diff = yhat - y
        mask = (diff.ge(0).float() - tau).detach()
        return (mask * diff).mean()


def augment(x, tau=None):
    """
    Augment input features with quantile information.
    
    Args:
        x: Input features as PyTorch tensor
        tau: Quantile level (between 0 and 1) as float or tensor
        
    Returns:
        Augmented features with quantile information
    """
    if tau is None:
        tau = torch.zeros(x.size(0), 1, device=x.device).fill_(0.5)
    elif isinstance(tau, float):
        tau = torch.zeros(x.size(0), 1, device=x.device).fill_(tau)
        
    return torch.cat((x, (tau - 0.5) * 12), 1)


class SimultaneousQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Neural network-based Simultaneous Quantile Regression (SQR) model.
    
    This model implements the SQR approach from "Single-Model Uncertainties for Deep Learning"
    which allows training a single neural network that can predict multiple quantiles.
    
    Parameters:
    -----------
    hidden_layers : list, default=[128, 128]
        List of integers specifying the number of neurons in each hidden layer.
    
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.
    
    weight_decay : float, default=1e-2
        L2 regularization parameter. Default matches original paper implementation.
    
    n_epochs : int, default=1000
        Number of training epochs. Default matches original paper implementation.
    
    batch_size : int, default=None
        Batch size for training. If None, process all data at once (as in original implementation).
        For large datasets, using a batch_size=64 or 128 can significantly improve performance.
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    quantile : float, default=None
        The quantile to predict (between 0 and 1). If None, the model can predict
        any quantile at inference time.
        
    validation_split : float, default=0.0
        Fraction of training data to use for validation during early stopping.
        Set to 0 to disable early stopping (as in original implementation).
        
    patience : int, default=300
        Number of epochs to wait for improvement before stopping training.
        
    min_delta : float, default=1e-6
        Minimum improvement required to consider as improvement.
        
    verbose : int, default=0
        Verbosity level. 0 = silent, 1 = periodic loss updates, 2 = one line per epoch.
        
    device : str, default='auto'
        Device to use for training ('cpu', 'cuda', or 'auto' to automatically detect).
        
    use_amp : bool, default=True
        Whether to use automatic mixed precision training for faster performance on GPUs.
    """
    def __init__(self, hidden_layers=[128, 128], learning_rate=1e-3, weight_decay=1e-2,
                 n_epochs=1000, batch_size=None, random_state=None, quantile=None,
                 validation_split=0.0, patience=300, min_delta=1e-6, verbose=0,
                 device='auto', use_amp=True):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.quantile = quantile
        self.validation_split = validation_split
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.device = device
        self.use_amp = use_amp
        
    def _build_network(self, input_dim):
        """Build the neural network architecture."""
        # Match the structure of the original sqr.py network
        network = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, self.hidden_layers[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layers[0], self.hidden_layers[-1] if len(self.hidden_layers) > 1 else self.hidden_layers[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layers[-1], 1)
        )
        return network
    
    def fit(self, X, y, quantile=None):
        """
        Fit the SQR model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or torch.Tensor
            Training data.
        
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.
        
        quantile : float, default=None
            The quantile to predict (between 0 and 1). 
            If provided, this overrides the quantile from initialization.
            If None and self.quantile is None, the model is trained for all quantiles.
        
        Returns:
        --------
        self : object
            Returns self.
        """
        # Set seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # If quantile is provided in fit, it overrides the one from initialization
        if quantile is not None:
            self.quantile = quantile
        
        # Determine device
        if self.device == 'auto':
            self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device_ = self.device
            
        if self.verbose > 0:
            print(f"Using device: {self.device_}")
            
        # Handle different input types - keeping tensors on CPU initially
        if isinstance(X, np.ndarray) or isinstance(X, list):
            # Convert numpy arrays to PyTorch tensors (on CPU)
            X_tensor = torch.FloatTensor(X)
        else:
            # Already a PyTorch tensor - make sure it's on CPU
            X_tensor = X.cpu() if hasattr(X, 'cpu') else X
            
        if isinstance(y, np.ndarray) or isinstance(y, list):
            # For numpy arrays, ensure it's the right shape for PyTorch
            if len(y.shape) == 1 or y.shape[1] == 1:
                y_tensor = torch.FloatTensor(y).view(-1, 1)
            else:
                y_tensor = torch.FloatTensor(y)
        else:
            # Already a PyTorch tensor - make sure it's on CPU
            y_tensor = (y.cpu() if hasattr(y, 'cpu') else y).view(-1, 1) if y.dim() == 1 else y
        
        # Store the number of features for future use
        self.n_features_in_ = X_tensor.shape[1]
        
        # Handle validation split if requested
        use_early_stopping = self.validation_split > 0
        if use_early_stopping:
            # Use PyTorch's random_split for reproducible splitting
            dataset_size = len(X_tensor)
            val_size = int(dataset_size * self.validation_split)
            train_size = dataset_size - val_size
            
            indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(self.random_state))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            X_train_tensor = X_tensor[train_indices]
            y_train_tensor = y_tensor[train_indices]
            X_val_tensor = X_tensor[val_indices]
            y_val_tensor = y_tensor[val_indices]
        else:
            # No validation split - use all data for training
            X_train_tensor = X_tensor
            y_train_tensor = y_tensor
            X_val_tensor, y_val_tensor = None, None
        
        # Create DataLoader for batched training if batch_size is specified
        use_batched_training = self.batch_size is not None and self.batch_size < len(X_train_tensor)
        if use_batched_training:
            # Create TensorDataset and DataLoader while tensors are still on CPU
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            # Determine appropriate number of workers for DataLoader
            num_workers = 0
            if self.device_ == 'cuda':
                try:
                    # Use more workers on CUDA for parallel data loading
                    import multiprocessing
                    num_workers = min(4, multiprocessing.cpu_count() // 2)  # Use up to 4 workers, but not more than half the CPUs
                except ImportError:
                    num_workers = 0
                    
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=self.device_ == 'cuda',  # Pin memory only if using GPU
                num_workers=num_workers,  # Use multiple workers for parallel data loading on GPU
                persistent_workers=num_workers > 0  # Keep workers alive between epochs
            )
        
        # Move validation tensors to the device after DataLoader creation
        if use_early_stopping and X_val_tensor is not None:
            X_val_tensor = X_val_tensor.to(self.device_)
            y_val_tensor = y_val_tensor.to(self.device_)
        
        # Move non-batched training tensors to device if not using DataLoader
        if not use_batched_training:
            X_train_tensor = X_train_tensor.to(self.device_)
            y_train_tensor = y_train_tensor.to(self.device_)
        
        # Build the network and move to device
        self.network_ = self._build_network(self.n_features_in_).to(self.device_)
        
        # Define optimizer and loss function
        self.optimizer_ = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Add learning rate scheduler for better convergence
        if self.n_epochs > 100:
            # Use OneCycleLR for faster convergence
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_,
                max_lr=self.learning_rate * 10,  # Peak at 10x the base lr
                steps_per_epoch=1 if not use_batched_training else len(train_loader),
                epochs=self.n_epochs,
                pct_start=0.3,  # Use 30% of iterations for warmup
                div_factor=10.0,  # Start with lr/10
                final_div_factor=1e4,  # End with lr/1000
                anneal_strategy='cos'  # Use cosine annealing
            )
        else:
            # For short training, just use constant LR
            scheduler = None
            
        self.loss_fn_ = QuantileLoss()
        
        # Initialize mixed precision training if requested and available
        use_amp = self.use_amp and self.device_ == 'cuda' and hasattr(torch, 'amp')
        if use_amp:
            # Use the newer API: torch.amp instead of torch.cuda.amp
            self.scaler_ = torch.amp.GradScaler()
            if self.verbose > 0:
                print("Using mixed precision training")
        
        # Initialize variables for early stopping if needed
        if use_early_stopping:
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            self.val_losses_ = []
        
        # Keep track of training losses
        self.train_losses_ = []
        
        # Decide how to display progress based on verbose setting
        if self.verbose == 1 and use_early_stopping:
            # Create a more descriptive label for the progress bar
            if self.quantile is None:
                desc = f"Training SQR (all quantiles)"
            else:
                desc = f"Training SQR (τ={self.quantile})"
            
            # Use auto_tqdm which chooses the appropriate tqdm version
            epoch_iterator = auto_tqdm(range(self.n_epochs), desc=desc, leave=True)
        else:
            # Don't use progress bar, just iterate through epochs
            epoch_iterator = range(self.n_epochs)
            if self.verbose > 0:
                # Print a header for the simple output
                print(f"Training SQR model{' for τ=' + str(self.quantile) if self.quantile is not None else ' for all quantiles'}")
                if self.verbose == 1:
                    # Print every 100 epochs in simple mode
                    print_interval = 20 if self.device_ == 'cuda' else 100
                else:
                    # Print every epoch in verbose mode
                    print_interval = 1
        
        # Training loop
        self.network_.train()
        for epoch in epoch_iterator:
            # Training step
            train_loss = 0.0
            batches = 0
            
            # Use mini-batches if batch_size is specified
            if use_batched_training:
                for X_batch, y_batch in train_loader:
                    # Move batch data to the device
                    X_batch = X_batch.to(self.device_)
                    y_batch = y_batch.to(self.device_)
                    
                    # Generate random quantiles for training or use fixed quantile
                    if self.quantile is None:
                        taus = torch.rand(len(X_batch), 1, device=self.device_)
                    else:
                        taus = torch.zeros(len(X_batch), 1, device=self.device_).fill_(self.quantile)
                    
                    # Forward pass with mixed precision if enabled
                    self.optimizer_.zero_grad()
                    
                    if use_amp:
                        with torch.amp.autocast(device_type=self.device_):
                            y_pred = self.network_(augment(X_batch, taus))
                            loss = self.loss_fn_(y_pred, y_batch, taus)
                        
                        # Backward pass with gradient scaling
                        self.scaler_.scale(loss).backward()
                        self.scaler_.step(self.optimizer_)
                        self.scaler_.update()
                    else:
                        # Standard forward and backward pass
                        y_pred = self.network_(augment(X_batch, taus))
                        loss = self.loss_fn_(y_pred, y_batch, taus)
                        loss.backward()
                        self.optimizer_.step()
                    
                    # Step the scheduler if it exists
                    if scheduler is not None:
                        scheduler.step()
                    
                    train_loss += loss.item()
                    batches += 1
                    
                    # Print loss for current batch if verbose level is high enough
                    if self.verbose >= 2:
                        print(f"Epoch {epoch+1} Batch {batches} - Loss: {loss.item():.6f}")
            else:
                # Process the entire dataset at once - just like in the original implementation
                if self.quantile is None:
                    taus = torch.rand(len(X_train_tensor), 1, device=self.device_)
                else:
                    taus = torch.zeros(len(X_train_tensor), 1, device=self.device_).fill_(self.quantile)
                
                # Forward pass with mixed precision if enabled
                self.optimizer_.zero_grad()
                
                if use_amp:
                    with torch.amp.autocast(device_type=self.device_):
                        y_pred = self.network_(augment(X_train_tensor, taus))
                        loss = self.loss_fn_(y_pred, y_train_tensor, taus)
                    
                    # Backward pass with gradient scaling
                    self.scaler_.scale(loss).backward()
                    self.scaler_.step(self.optimizer_)
                    self.scaler_.update()
                else:
                    # Standard forward and backward pass
                    y_pred = self.network_(augment(X_train_tensor, taus))
                    loss = self.loss_fn_(y_pred, y_train_tensor, taus)
                    loss.backward()
                    self.optimizer_.step()
                
                # Step the scheduler if it exists and not using batches
                if scheduler is not None and not use_batched_training:
                    scheduler.step()
                
                train_loss = loss.item()
                batches = 1
            
            # Calculate average training loss for this epoch
            avg_train_loss = train_loss / batches
            self.train_losses_.append(avg_train_loss)
            
            # Print training loss at intervals if verbose > 0 and not using tqdm
            if self.verbose > 0 and not isinstance(epoch_iterator, auto_tqdm) and epoch % print_interval == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {avg_train_loss:.6f}")
            
            # Validation and early stopping logic (only if validation split > 0)
            if use_early_stopping and X_val_tensor is not None:
                self.network_.eval()
                with torch.no_grad():
                    if self.quantile is None:
                        # For "all quantiles" mode, evaluate on multiple quantiles
                        val_losses = []
                        for q in [0.1, 0.5, 0.9]:  # Evaluate on these quantiles
                            taus_val = torch.zeros(len(X_val_tensor), 1, device=self.device_).fill_(q)
                            y_val_pred = self.network_(augment(X_val_tensor, taus_val))
                            val_loss = self.loss_fn_(y_val_pred, y_val_tensor, taus_val).item()
                            val_losses.append(val_loss)
                        val_loss = np.mean(val_losses)
                    else:
                        # For single quantile mode
                        taus_val = torch.zeros(len(X_val_tensor), 1, device=self.device_).fill_(self.quantile)
                        y_val_pred = self.network_(augment(X_val_tensor, taus_val))
                        val_loss = self.loss_fn_(y_val_pred, y_val_tensor, taus_val).item()
                    
                    self.val_losses_.append(val_loss)
                self.network_.train()
                
                # Print validation loss if verbose
                if self.verbose > 0 and not isinstance(epoch_iterator, auto_tqdm) and epoch % print_interval == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")
                    
                # Check for improvement
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model state
                    best_model_state = {key: value.cpu().clone() for key, value in self.network_.state_dict().items()}
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= self.patience:
                    if self.verbose > 0:
                        # Close the progress bar properly before printing early stopping message
                        if isinstance(epoch_iterator, auto_tqdm) and hasattr(epoch_iterator, 'close'):
                            epoch_iterator.close()
                        print(f"Early stopping at epoch {epoch+1}/{self.n_epochs} - Best validation loss: {best_val_loss:.6f}")
                    break
        
        # Update progress bar to 100% when training is complete if not interrupted by early stopping
        if isinstance(epoch_iterator, auto_tqdm) and hasattr(epoch_iterator, 'close'):
            epoch_iterator.close()
            
        # Print final loss
        if self.verbose > 0 and not use_early_stopping:
            print(f"Training completed - Final Loss: {self.train_losses_[-1]:.6f}")
            
        # Load the best model if early stopping was used
        if use_early_stopping and best_model_state is not None:
            self.network_.load_state_dict(best_model_state)
        
        # Mark the model as fitted
        self.fitted_ = True
        return self
    
    def predict(self, X, quantile=None):
        """
        Predict using the SQR model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or torch.Tensor
            Samples to predict for.
        
        quantile : float, default=None
            The quantile to predict (between 0 and 1). 
            If provided, this overrides both the quantile from fit and from initialization.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,) or torch.Tensor
            Predicted values.
        """
        # Check if model is fitted
        check_is_fitted(self, attributes=["fitted_", "network_"])
        
        # Determine which quantile to use
        if quantile is not None:
            tau = quantile
        elif self.quantile is not None:
            tau = self.quantile
        else:
            tau = 0.5  # Default to median if no quantile is specified
        
        # Track whether to return numpy or tensor
        input_is_tensor = isinstance(X, torch.Tensor)
        return_numpy = not input_is_tensor
        
        # Convert input to PyTorch tensor if it's not already
        if not input_is_tensor:
            # Use sklearn's check_array for numpy arrays
            X_checked = check_array(X)
            X_tensor = torch.FloatTensor(X_checked)
        else:
            # For PyTorch tensors, skip sklearn validation
            X_tensor = X
        
        # Move input tensor to the appropriate device
        original_device = X_tensor.device if input_is_tensor else None
        X_tensor = X_tensor.to(self.device_)
        
        # Set model to evaluation mode
        self.network_.eval()
        
        # Make prediction
        with torch.no_grad():
            # Create quantile tensor
            if isinstance(tau, float):
                tau_tensor = torch.zeros(len(X_tensor), 1, device=self.device_).fill_(tau)
            else:
                tau_tensor = tau.to(self.device_) if hasattr(tau, 'to') else tau
                
            # Apply augmentation exactly as in original implementation
            X_augmented = augment(X_tensor, tau_tensor)
            
            # Get predictions
            y_pred = self.network_(X_augmented).squeeze()
            
            # Return to original device if input was tensor
            if input_is_tensor and original_device is not None:
                y_pred = y_pred.to(original_device)
            # Convert to numpy if input was numpy
            elif return_numpy:
                y_pred = y_pred.cpu().numpy()
        
        return y_pred
    
    def predict_base(self, X):
        """
        Base prediction method for compatibility with other models in the CLEAR framework.
        This is just an alias for predict with the median quantile.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        return self.predict(X, quantile=0.5)
    
    def plot_losses(self):
        """Plot the training and validation losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses_, label='Training Loss')
        if hasattr(self, 'val_losses_') and len(self.val_losses_) > 0:
            plt.plot(self.val_losses_, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        return plt


# For backward compatibility and simpler usage
SQR = SimultaneousQuantileRegressor


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    print("=" * 80)
    print("Simultaneous Quantile Regression (SQR) Example")
    print("=" * 80)
    
    # Generate synthetic data exactly like in the original sqr.py
    torch.manual_seed(42)  # For reproducibility
    
    n = 1000
    d = 5
    
    # Use PyTorch tensors for data generation to exactly match the original implementation
    x = torch.randn(n, d)
    y = x[:, 0].view(-1, 1).mul(5).cos() + 0.3 * torch.randn(n, 1)
    
    # Configure faster training with GPU - use larger batch size and fewer epochs for even faster training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 10000 if device == 'cuda' else 1000  # Fewer epochs for CPU
    batch_size = 256 if device == 'cuda' else None
    
    print(f"\nUsing device: {device}")
    print(f"Training for {n_epochs} epochs with batch size: {batch_size}")
    
    # Try to use torch.compile if available (PyTorch 2.0+)
    use_compile = hasattr(torch, 'compile')
    if use_compile:
        print("Using torch.compile for faster training")
    
    # Time the training
    start_time = time.time()
    
    print("\n1. Training a joint model for all quantiles...")
    # Train a joint model for all quantiles with GPU acceleration
    sqr_all = SimultaneousQuantileRegressor(
        hidden_layers=[256, 256],
        learning_rate=1e-3,
        weight_decay=1e-2,
        n_epochs=n_epochs,
        batch_size=batch_size,
        quantile=None,             # Train for all quantiles
        random_state=42,
        validation_split=0.0,      # No validation split - train on all data
        verbose=1,                 # Simple loss output with no progress bar
        device=device,             # Use GPU if available
        use_amp=True               # Use mixed precision for faster training
    )
    
    # Compile the model if available
    if use_compile:
        sqr_all.network_ = torch.compile(sqr_all._build_network(x.shape[1])).to(device)
    
    sqr_all.fit(x, y)  # Pass PyTorch tensors directly
    
    print("\n2. Training separate models for specific quantiles...")
    print("\nTraining model for τ=0.1...")
    # Train separate models for specific quantiles
    sqr_01 = SimultaneousQuantileRegressor(
        hidden_layers=[128, 128],
        learning_rate=1e-3,
        weight_decay=1e-2,
        n_epochs=n_epochs,
        batch_size=batch_size,
        quantile=0.1, 
        random_state=42,
        validation_split=0.0,
        verbose=1,
        device=device,
        use_amp=True
    )
    
    # Compile the model if available
    if use_compile:
        sqr_01.network_ = torch.compile(sqr_01._build_network(x.shape[1])).to(device)
        
    sqr_01.fit(x, y)  # Pass PyTorch tensors directly
    
    print("\nTraining model for τ=0.5...")
    sqr_50 = SimultaneousQuantileRegressor(
        hidden_layers=[128, 128],
        learning_rate=1e-3,
        weight_decay=1e-2,
        n_epochs=n_epochs,
        batch_size=batch_size,
        quantile=0.5, 
        random_state=42,
        validation_split=0.0,
        verbose=1,
        device=device,
        use_amp=True
    )
    
    # Compile the model if available
    if use_compile:
        sqr_50.network_ = torch.compile(sqr_50._build_network(x.shape[1])).to(device)
        
    sqr_50.fit(x, y)
    
    print("\nTraining model for τ=0.9...")
    sqr_90 = SimultaneousQuantileRegressor(
        hidden_layers=[128, 128],
        learning_rate=1e-3,
        weight_decay=1e-2, 
        n_epochs=n_epochs,
        batch_size=batch_size,
        quantile=0.9, 
        random_state=42,
        validation_split=0.0,
        verbose=1,
        device=device,
        use_amp=True
    )
    
    # Compile the model if available
    if use_compile:
        sqr_90.network_ = torch.compile(sqr_90._build_network(x.shape[1])).to(device)
        
    sqr_90.fit(x, y)
    
    # Report total training time
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    
    # Generate new test data, exactly as in original sqr.py
    print("\n3. Generating test data and making predictions...")
    x_test = torch.randn(n, d)
    y_test = x_test[:, 0].view(-1, 1).mul(5).cos() + 0.3 * torch.randn(n, 1)
    
    # Sort by first feature for better visualization - exactly as in original
    o = torch.sort(x_test[:, 0])[1]
    
    # Create visualization matching sqr.py exactly
    print("\n4. Creating visualizations...")
    
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)
    plt.rc('text.latex', preamble=r'\usepackage{times}')
    plt.figure(figsize=(15, 4))
    plt.rc('font', family='serif')
    
    # Plot separate models
    plt.subplot(1, 4, 1)
    plt.title("separate estimation")
    plt.plot(x_test[o, 0].cpu().numpy(), y_test[o].cpu().detach().numpy(), '.')
    plt.plot(x_test[o, 0].cpu().numpy(), sqr_01.predict(x_test[o]), alpha=0.75, label="$\\tau_{0.1}$")
    plt.plot(x_test[o, 0].cpu().numpy(), sqr_50.predict(x_test[o]), alpha=0.75, label="$\\tau_{0.5}$")
    plt.plot(x_test[o, 0].cpu().numpy(), sqr_90.predict(x_test[o]), alpha=0.75, label="$\\tau_{0.9}$")
    plt.legend()
    
    # Plot differences between separate models
    plt.subplot(1, 4, 2)
    plt.plot(x_test[o, 0].cpu().numpy(), 
             sqr_90.predict(x_test[o]) - sqr_50.predict(x_test[o]), 
             alpha=0.75, label="$\\tau_{0.9} - \\tau_{0.5}$")
    plt.plot(x_test[o, 0].cpu().numpy(), 
             sqr_50.predict(x_test[o]) - sqr_01.predict(x_test[o]), 
             alpha=0.75, label="$\\tau_{0.5} - \\tau_{0.1}$")
    plt.axhline(0, ls="--", color="gray")
    plt.legend()
    
    # Plot joint model
    plt.subplot(1, 4, 3)
    plt.title("joint estimation")
    plt.plot(x_test[o, 0].cpu().numpy(), y_test[o].cpu().detach().numpy(), '.')
    plt.plot(x_test[o, 0].cpu().numpy(), sqr_all.predict(x_test[o], quantile=0.1), alpha=0.75, label="$\\tau_{0.1}$")
    plt.plot(x_test[o, 0].cpu().numpy(), sqr_all.predict(x_test[o], quantile=0.5), alpha=0.75, label="$\\tau_{0.5}$")
    plt.plot(x_test[o, 0].cpu().numpy(), sqr_all.predict(x_test[o], quantile=0.9), alpha=0.75, label="$\\tau_{0.9}$")
    plt.legend()
    
    # Plot differences between joint model quantiles
    plt.subplot(1, 4, 4)
    plt.plot(x_test[o, 0].cpu().numpy(), 
             sqr_all.predict(x_test[o], quantile=0.9) - sqr_all.predict(x_test[o], quantile=0.5), 
             alpha=0.75, label="$\\tau_{0.9} - \\tau_{0.5}$")
    plt.plot(x_test[o, 0].cpu().numpy(), 
             sqr_all.predict(x_test[o], quantile=0.5) - sqr_all.predict(x_test[o], quantile=0.1), 
             alpha=0.75, label="$\\tau_{0.5} - \\tau_{0.1}$")
    plt.axhline(0, ls="--", color="gray")
    plt.legend()
    
    # Use tight_layout with keyword arguments instead of positional arguments
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig("sqr_comparison.png")
    plt.show()
    print("Comparison plot saved to 'sqr_comparison.png'")
    
    # Check if quantiles cross (this shouldn't happen with a good model)
    print("\n5. Checking quantile crossing issues:")
    
    # Predict using the models - ensure all are numpy arrays
    preds_01_separate = sqr_01.predict(x_test).cpu().numpy() if torch.is_tensor(sqr_01.predict(x_test)) else sqr_01.predict(x_test)
    preds_50_separate = sqr_50.predict(x_test).cpu().numpy() if torch.is_tensor(sqr_50.predict(x_test)) else sqr_50.predict(x_test)
    preds_90_separate = sqr_90.predict(x_test).cpu().numpy() if torch.is_tensor(sqr_90.predict(x_test)) else sqr_90.predict(x_test)
    
    preds_01_joint = sqr_all.predict(x_test, quantile=0.1).cpu().numpy() if torch.is_tensor(sqr_all.predict(x_test, quantile=0.1)) else sqr_all.predict(x_test, quantile=0.1)
    preds_50_joint = sqr_all.predict(x_test, quantile=0.5).cpu().numpy() if torch.is_tensor(sqr_all.predict(x_test, quantile=0.5)) else sqr_all.predict(x_test, quantile=0.5)
    preds_90_joint = sqr_all.predict(x_test, quantile=0.9).cpu().numpy() if torch.is_tensor(sqr_all.predict(x_test, quantile=0.9)) else sqr_all.predict(x_test, quantile=0.9)
    
    # Count crossing issues
    crossing_01_50_separate = np.sum(preds_01_separate > preds_50_separate)
    crossing_50_90_separate = np.sum(preds_50_separate > preds_90_separate)
    crossing_01_50_joint = np.sum(preds_01_joint > preds_50_joint)
    crossing_50_90_joint = np.sum(preds_50_joint > preds_90_joint)
    
    print(f"Separate models - crossings between 0.1 and 0.5: {crossing_01_50_separate}/{len(x_test)}")
    print(f"Separate models - crossings between 0.5 and 0.9: {crossing_50_90_separate}/{len(x_test)}")
    print(f"Joint model - crossings between 0.1 and 0.5: {crossing_01_50_joint}/{len(x_test)}")
    print(f"Joint model - crossings between 0.5 and 0.9: {crossing_50_90_joint}/{len(x_test)}")

    print("\n" + "=" * 80)
    print("Quantile GAM (QuantileGAM) Example")
    print("=" * 80)

    # Generate some 1D data for GAM demonstration
    np.random.seed(123)
    n_gam = 200
    X_gam = np.sort(np.random.rand(n_gam, 1) * 10, axis=0)
    y_gam_true = np.sin(X_gam[:,0]) * X_gam[:,0]/3 + 5
    y_gam_noise = y_gam_true + np.random.normal(0, 0.5 + 0.2 * X_gam[:,0], n_gam) # Heteroscedastic noise

    # Fit QuantileGAM for different quantiles
    qgam_01 = QuantileGAM(quantile=0.1, n_splines=10, lam=0.1)
    qgam_01.fit(X_gam, y_gam_noise)
    pred_gam_01 = qgam_01.predict(X_gam)

    qgam_50 = QuantileGAM(quantile=0.5, n_splines=10, lam=0.1)
    qgam_50.fit(X_gam, y_gam_noise)
    pred_gam_50 = qgam_50.predict(X_gam)

    qgam_90 = QuantileGAM(quantile=0.9, n_splines=10, lam=0.1)
    qgam_90.fit(X_gam, y_gam_noise)
    pred_gam_90 = qgam_90.predict(X_gam)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_gam, y_gam_noise, label='Data', color='black', s=10, alpha=0.5)
    plt.plot(X_gam, y_gam_true, label='True Function', color='gray', linestyle='--')
    plt.plot(X_gam, pred_gam_01, label='QuantileGAM (q=0.1)', color='blue')
    plt.plot(X_gam, pred_gam_50, label='QuantileGAM (q=0.5)', color='red')
    plt.plot(X_gam, pred_gam_90, label='QuantileGAM (q=0.9)', color='green')
    plt.title('QuantileGAM Demonstration')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    try:
        plt.savefig("qgam_demonstration.png")
        print("\nQuantileGAM demonstration plot saved to qgam_demonstration.png")
    except Exception as e:
        print(f"\nCould not save QuantileGAM plot: {e}. Displaying instead.")
        plt.show()

    print("\nCLEAR Models module demonstration finished.") 