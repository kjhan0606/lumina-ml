#!/usr/bin/env python3
"""Train the MLP spectral emulator with feature-weighted composite loss.

v2: Uses FeatureWeightedLoss (MSE_pca + feature windows + UV ratio).

Usage:
  python3 scripts/03_train_emulator.py
  python3 scripts/03_train_emulator.py --epochs 2000 --lr 5e-4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumina_ml import config as cfg
from lumina_ml.emulator import SpectralMLP, EmulatorTrainer
from lumina_ml.preprocessing import SpectralPCA, asinh_inverse, measure_spectral_features


def main():
    parser = argparse.ArgumentParser(description="Train the spectral emulator")
    parser.add_argument('--epochs', type=int, default=cfg.NN_MAX_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=cfg.NN_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=cfg.NN_LEARNING_RATE)
    parser.add_argument('--patience', type=int, default=cfg.NN_PATIENCE)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--no-feature-loss', action='store_true',
                        help='Disable feature-weighted loss (use simple MSE)')
    args = parser.parse_args()

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load(str(cfg.DATA_PROCESSED / "X_train.npy"))
    X_val = np.load(str(cfg.DATA_PROCESSED / "X_val.npy"))
    Y_train = np.load(str(cfg.DATA_PROCESSED / "Y_train.npy"))
    Y_val = np.load(str(cfg.DATA_PROCESSED / "Y_val.npy"))

    n_pca = Y_train.shape[1]
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Input dim: {cfg.N_PARAMS}, Output dim (PCA): {n_pca}")

    # Load PCA model for feature-weighted loss
    pca = SpectralPCA.load(cfg.DATA_PROCESSED / "pca_model.pkl")

    # Create model
    model = SpectralMLP(n_input=cfg.N_PARAMS, n_output=n_pca)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"  Hidden layers: {cfg.NN_HIDDEN_LAYERS}")
    print(f"  Activation: SiLU, Dropout: {cfg.NN_DROPOUT}")

    # Create trainer with composite loss
    use_pca = None if args.no_feature_loss else pca
    if use_pca is not None:
        print(f"\n  Composite loss enabled:")
        print(f"    lambda_feat = {cfg.FEATURE_LOSS_LAMBDA}")
        print(f"    lambda_uv   = {cfg.UV_RATIO_LOSS_LAMBDA}")
        print(f"    Feature windows: {list(cfg.FEATURE_WINDOWS.keys())}")
    else:
        print(f"\n  Simple MSE loss (no feature weighting)")

    # Train
    print(f"\nTraining (max {args.epochs} epochs, patience {args.patience})...")
    trainer = EmulatorTrainer(model, device=args.device, pca_model=use_pca)

    # Re-create optimizer and scheduler with CLI args
    import torch
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=cfg.NN_WEIGHT_DECAY,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer, T_0=cfg.NN_T0, T_mult=cfg.NN_T_MULT,
    )

    history = trainer.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience,
    )

    # Save model
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = cfg.MODELS_DIR / "emulator.pt"
    trainer.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Validation metrics
    print("\n--- Validation ---")
    model.eval()
    device = trainer.device
    with torch.no_grad():
        xv = torch.FloatTensor(X_val).to(device)
        pred = model(xv).cpu().numpy()

    # PCA-space error
    mse_pca = np.mean((pred - Y_val)**2)
    print(f"  PCA MSE (standardized): {mse_pca:.6f}")

    # Spectrum-space error (asinh)
    spectra_val = np.load(str(cfg.DATA_PROCESSED / "spectra_val.npy"))
    spectra_pred = pca.inverse_transform(pred)
    rms_per_spec = np.sqrt(np.mean((spectra_pred - spectra_val)**2, axis=1))
    print(f"  Spectrum RMS (asinh): mean={rms_per_spec.mean():.4f}, "
          f"median={np.median(rms_per_spec):.4f}, max={rms_per_spec.max():.4f}")

    # Linear flux RMS
    spec_linear_true = asinh_inverse(spectra_val)
    spec_linear_pred = asinh_inverse(spectra_pred)
    rms_linear = np.sqrt(np.mean((spec_linear_true - spec_linear_pred)**2, axis=1))
    print(f"  Spectrum RMS (linear): mean={rms_linear.mean():.4f}, "
          f"median={np.median(rms_linear):.4f}, max={rms_linear.max():.4f}")

    # Feature-specific errors
    grid = cfg.SPECTRUM_GRID
    print(f"\n  Feature-specific RMS (asinh):")
    for name, (wmin, wmax) in cfg.FEATURE_WINDOWS.items():
        mask = (grid >= wmin) & (grid <= wmax)
        if mask.sum() > 0:
            f_rms = np.sqrt(np.mean((spectra_pred[:, mask] - spectra_val[:, mask])**2, axis=1))
            print(f"    {name:15s}: mean={f_rms.mean():.4f}  max={f_rms.max():.4f}")

    # UV/optical ratio accuracy
    uv_mask = (grid >= cfg.UV_BAND[0]) & (grid <= cfg.UV_BAND[1])
    opt_mask = (grid >= cfg.OPT_BAND[0]) & (grid <= cfg.OPT_BAND[1])
    ratio_true = spec_linear_true[:, uv_mask].mean(axis=1) / (spec_linear_true[:, opt_mask].mean(axis=1) + 1e-30)
    ratio_pred = spec_linear_pred[:, uv_mask].mean(axis=1) / (spec_linear_pred[:, opt_mask].mean(axis=1) + 1e-30)
    ratio_err = np.abs(ratio_true - ratio_pred)
    print(f"\n  UV/opt ratio error: mean={ratio_err.mean():.4f}, max={ratio_err.max():.4f}")

    # Si II velocity and depth accuracy (sample)
    print(f"\n  Si II feature accuracy (100 random val samples):")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(spectra_val), min(100, len(spectra_val)), replace=False)
    vel_errs, depth_errs = [], []
    for idx in sample_idx:
        feat_true = measure_spectral_features(spectra_val[idx], is_asinh=True)
        feat_pred = measure_spectral_features(spectra_pred[idx], is_asinh=True)
        vel_errs.append(abs(feat_true['si_ii_vel'] - feat_pred['si_ii_vel']))
        depth_errs.append(abs(feat_true['si_ii_depth'] - feat_pred['si_ii_depth']))
    print(f"    Si II velocity error: mean={np.mean(vel_errs):.0f} km/s, "
          f"max={np.max(vel_errs):.0f} km/s")
    print(f"    Si II depth error:    mean={np.mean(depth_errs):.4f}, "
          f"max={np.max(depth_errs):.4f}")

    # Save training history
    np.savez(str(cfg.MODELS_DIR / "training_history.npz"), **history)

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss
        ax = axes[0, 0]
        ax.semilogy(history['train_loss'], label='Train', alpha=0.7)
        ax.semilogy(history['val_loss'], label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Composite Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PCA loss
        ax = axes[0, 1]
        ax.semilogy(history['train_pca'], label='Train PCA', alpha=0.7)
        ax.semilogy(history['val_pca'], label='Val PCA', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PCA MSE')
        ax.set_title('PCA Coefficient Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Feature loss
        ax = axes[1, 0]
        if max(history.get('train_feat', [0])) > 0:
            ax.semilogy(history['train_feat'], label='Train Feature', alpha=0.7)
            ax.semilogy(history['val_feat'], label='Val Feature', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Feature Loss')
        ax.set_title('Feature Window Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 1]
        ax.plot(history['lr'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = cfg.MODELS_DIR / "training_curves.png"
        plt.savefig(str(fig_path), dpi=150)
        print(f"\n  Training curves saved to {fig_path}")
        plt.close()
    except Exception as e:
        print(f"  (Plotting failed: {e})")

    print("\nDone!")


if __name__ == '__main__':
    main()
