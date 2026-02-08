"""PyTorch MLP spectral emulator: 15D params -> PCA coefficients.

v2: Feature-weighted composite loss for better absorption line accuracy.
  Total Loss = MSE_pca + lambda_feat * L_features + lambda_uv * L_uv_ratio
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as cfg


class SpectralMLP(nn.Module):
    """Multi-Layer Perceptron: maps normalized params to standardized PCA coefficients."""

    def __init__(self, n_input: int = cfg.N_PARAMS,
                 n_output: int = 50,
                 hidden_layers: list = None,
                 dropout: float = cfg.NN_DROPOUT):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = cfg.NN_HIDDEN_LAYERS

        layers = []
        prev = n_input
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_output))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureWeightedLoss(nn.Module):
    """Composite loss: MSE on PCA + feature-weighted spectrum loss + UV ratio loss.

    Total = MSE_pca(pred, target)
          + lambda_feat * sum_features[ MSE(pred_spectrum[window], target_spectrum[window]) ]
          + lambda_uv * MSE(uv_ratio_pred, uv_ratio_target)
    """

    def __init__(self, pca_model, grid: np.ndarray = None,
                 lambda_feat: float = None, lambda_uv: float = None):
        super().__init__()

        self.lambda_feat = lambda_feat if lambda_feat is not None else cfg.FEATURE_LOSS_LAMBDA
        self.lambda_uv = lambda_uv if lambda_uv is not None else cfg.UV_RATIO_LOSS_LAMBDA

        if grid is None:
            grid = cfg.SPECTRUM_GRID

        # Pre-compute PCA inverse transform as tensors (for spectrum reconstruction)
        # PCA: spectrum = coeffs @ components + mean
        # Scaler: coeffs = coeffs_std * scale + center
        components = torch.FloatTensor(pca_model.pca.components_)       # [n_pca, n_bins]
        pca_mean = torch.FloatTensor(pca_model.pca.mean_)               # [n_bins]
        scaler_scale = torch.FloatTensor(pca_model.scaler.scale_)       # [n_pca]
        scaler_mean = torch.FloatTensor(pca_model.scaler.mean_)         # [n_pca]

        self.register_buffer('components', components)
        self.register_buffer('pca_mean', pca_mean)
        self.register_buffer('scaler_scale', scaler_scale)
        self.register_buffer('scaler_mean', scaler_mean)

        # Pre-compute feature window masks
        self.feature_masks = {}
        for name, (wmin, wmax) in cfg.FEATURE_WINDOWS.items():
            mask = torch.BoolTensor((grid >= wmin) & (grid <= wmax))
            self.register_buffer(f'mask_{name}', mask)
            self.feature_masks[name] = f'mask_{name}'

        # UV and optical band masks for ratio
        uv_mask = torch.BoolTensor((grid >= cfg.UV_BAND[0]) & (grid <= cfg.UV_BAND[1]))
        opt_mask = torch.BoolTensor((grid >= cfg.OPT_BAND[0]) & (grid <= cfg.OPT_BAND[1]))
        self.register_buffer('uv_mask', uv_mask)
        self.register_buffer('opt_mask', opt_mask)

        self.n_features = len(cfg.FEATURE_WINDOWS)

    def _reconstruct_spectrum(self, coeffs_std):
        """Reconstruct spectrum from standardized PCA coefficients (differentiable)."""
        coeffs = coeffs_std * self.scaler_scale + self.scaler_mean
        return coeffs @ self.components + self.pca_mean

    def forward(self, pred_pca, target_pca):
        """Compute composite loss.

        Args:
            pred_pca: [B, n_pca] predicted standardized PCA coefficients
            target_pca: [B, n_pca] target standardized PCA coefficients
        """
        # 1. Standard PCA MSE
        loss_pca = nn.functional.mse_loss(pred_pca, target_pca)

        # 2. Feature-weighted spectrum loss
        if self.lambda_feat > 0:
            spec_pred = self._reconstruct_spectrum(pred_pca)
            spec_target = self._reconstruct_spectrum(target_pca)

            loss_feat = torch.tensor(0.0, device=pred_pca.device)
            for name, buf_name in self.feature_masks.items():
                mask = getattr(self, buf_name)
                if mask.sum() > 0:
                    diff = spec_pred[:, mask] - spec_target[:, mask]
                    loss_feat = loss_feat + (diff ** 2).mean()
            loss_feat = loss_feat / self.n_features
        else:
            loss_feat = torch.tensor(0.0, device=pred_pca.device)

        # 3. UV/optical ratio loss
        if self.lambda_uv > 0:
            spec_pred = spec_pred if self.lambda_feat > 0 else self._reconstruct_spectrum(pred_pca)
            spec_target = spec_target if self.lambda_feat > 0 else self._reconstruct_spectrum(target_pca)

            uv_pred = spec_pred[:, self.uv_mask].mean(dim=1)
            opt_pred = spec_pred[:, self.opt_mask].mean(dim=1)
            ratio_pred = uv_pred / (opt_pred + 1e-8)

            uv_target = spec_target[:, self.uv_mask].mean(dim=1)
            opt_target = spec_target[:, self.opt_mask].mean(dim=1)
            ratio_target = uv_target / (opt_target + 1e-8)

            loss_uv = nn.functional.mse_loss(ratio_pred, ratio_target)
        else:
            loss_uv = torch.tensor(0.0, device=pred_pca.device)

        total = loss_pca + self.lambda_feat * loss_feat + self.lambda_uv * loss_uv
        return total, loss_pca, loss_feat, loss_uv


class EmulatorTrainer:
    """Training harness for the spectral MLP emulator."""

    def __init__(self, model: SpectralMLP, device: str = 'auto',
                 pca_model=None):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.NN_LEARNING_RATE,
            weight_decay=cfg.NN_WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=cfg.NN_T0, T_mult=cfg.NN_T_MULT,
        )

        # Use composite loss if PCA model provided, else simple MSE
        if pca_model is not None:
            self.criterion = FeatureWeightedLoss(pca_model).to(self.device)
            self.use_composite_loss = True
        else:
            self.criterion = nn.MSELoss()
            self.use_composite_loss = False

        self.history = {
            'train_loss': [], 'val_loss': [], 'lr': [],
            'train_pca': [], 'val_pca': [],
            'train_feat': [], 'val_feat': [],
            'train_uv': [], 'val_uv': [],
        }

    def _compute_loss(self, pred, target):
        """Compute loss, returning (total, pca, feat, uv) tuple."""
        if self.use_composite_loss:
            return self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target)
            zero = torch.tensor(0.0)
            return loss, loss, zero, zero

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray,
            X_val: np.ndarray, Y_val: np.ndarray,
            epochs: int = cfg.NN_MAX_EPOCHS,
            batch_size: int = cfg.NN_BATCH_SIZE,
            patience: int = cfg.NN_PATIENCE,
            verbose: bool = True) -> dict:
        """Train the emulator."""
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(Y_train),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(Y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_total, train_pca, train_feat, train_uv = 0.0, 0.0, 0.0, 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                total, lpca, lfeat, luv = self._compute_loss(pred, yb)
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.NN_GRAD_CLIP)
                self.optimizer.step()
                n = xb.size(0)
                train_total += total.item() * n
                train_pca += lpca.item() * n
                train_feat += lfeat.item() * n
                train_uv += luv.item() * n
            n_train = len(train_ds)
            train_total /= n_train
            train_pca /= n_train
            train_feat /= n_train
            train_uv /= n_train

            # Validation
            self.model.eval()
            val_total, val_pca, val_feat, val_uv = 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    total, lpca, lfeat, luv = self._compute_loss(pred, yb)
                    n = xb.size(0)
                    val_total += total.item() * n
                    val_pca += lpca.item() * n
                    val_feat += lfeat.item() * n
                    val_uv += luv.item() * n
            n_val = len(val_ds)
            val_total /= n_val
            val_pca /= n_val
            val_feat /= n_val
            val_uv /= n_val

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_total)
            self.history['val_loss'].append(val_total)
            self.history['lr'].append(lr)
            self.history['train_pca'].append(train_pca)
            self.history['val_pca'].append(val_pca)
            self.history['train_feat'].append(train_feat)
            self.history['val_feat'].append(val_feat)
            self.history['train_uv'].append(train_uv)
            self.history['val_uv'].append(val_uv)

            # Early stopping on total loss
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1 or patience_counter == patience):
                feat_str = ""
                if self.use_composite_loss:
                    feat_str = f"  pca={val_pca:.4f}  feat={val_feat:.4f}  uv={val_uv:.4f}"
                print(f"  Epoch {epoch:4d}: train={train_total:.6f}  val={val_total:.6f}"
                      f"{feat_str}  lr={lr:.2e}  best={best_val_loss:.6f}  "
                      f"patience={patience_counter}/{patience}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self.history

    def save(self, filepath: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'n_input': self.model.net[0].in_features,
            'n_output': self.model.net[-1].out_features,
            'history': self.history,
        }, filepath)

    @classmethod
    def load(cls, filepath: Path, device: str = 'auto') -> 'EmulatorTrainer':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        model = SpectralMLP(
            n_input=checkpoint['n_input'],
            n_output=checkpoint['n_output'],
        )
        model.load_state_dict(checkpoint['model_state'])
        trainer = cls(model, device=device)
        trainer.history = checkpoint.get('history', {})
        return trainer


class Emulator:
    """High-level emulator: params -> spectrum.

    Combines ParamScaler, SpectralMLP, and SpectralPCA into a single callable.
    Outputs are in asinh space by default; use predict_spectrum_linear() for linear flux.
    """

    def __init__(self, model: SpectralMLP, param_scaler, spectral_pca, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.param_scaler = param_scaler
        self.spectral_pca = spectral_pca

    def predict_spectrum(self, params: np.ndarray) -> np.ndarray:
        """Predict spectrum from physical parameters (in asinh space).

        Args:
            params: [n_params] or [N, n_params] array of physical parameters

        Returns:
            spectrum: [n_bins] or [N, n_bins] predicted spectrum (asinh-transformed)
        """
        single = params.ndim == 1
        if single:
            params = params[np.newaxis, :]

        params_norm = self.param_scaler.transform(params)
        with torch.no_grad():
            x = torch.FloatTensor(params_norm).to(self.device)
            pca_coeffs_std = self.model(x).cpu().numpy()

        spectra = self.spectral_pca.inverse_transform(pca_coeffs_std)

        if single:
            return spectra[0]
        return spectra

    def predict_spectrum_linear(self, params: np.ndarray) -> np.ndarray:
        """Predict spectrum in linear (peak-normalized) flux space.

        Applies inverse asinh transform to convert from asinh space.
        """
        from .preprocessing import asinh_inverse
        spec_asinh = self.predict_spectrum(params)
        return asinh_inverse(spec_asinh)

    def predict_pca(self, params: np.ndarray) -> np.ndarray:
        """Predict standardized PCA coefficients."""
        single = params.ndim == 1
        if single:
            params = params[np.newaxis, :]

        params_norm = self.param_scaler.transform(params)
        with torch.no_grad():
            x = torch.FloatTensor(params_norm).to(self.device)
            result = self.model(x).cpu().numpy()

        if single:
            return result[0]
        return result

    @classmethod
    def load(cls, models_dir: Path, processed_dir: Path, device: str = 'auto') -> 'Emulator':
        """Load a trained emulator from saved files."""
        from .preprocessing import SpectralPCA, ParamScaler

        checkpoint = torch.load(models_dir / "emulator.pt", map_location='cpu', weights_only=False)
        model = SpectralMLP(
            n_input=checkpoint['n_input'],
            n_output=checkpoint['n_output'],
        )
        model.load_state_dict(checkpoint['model_state'])

        param_scaler = ParamScaler.load(processed_dir / "param_scaler.pkl")
        spectral_pca = SpectralPCA.load(processed_dir / "pca_model.pkl")

        return cls(model, param_scaler, spectral_pca, device=device)
