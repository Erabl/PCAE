import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.loader import DataLoader
#from autoencoder import PointCloudAutoencoder
from ds_model import PointCloudAutoencoder
from pytorch_lightning.callbacks import ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback


import utils
import datetime

def objective(trial):
    # --- Sample hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    latent_dim = trial.suggest_int("latent_dim", 8, 256, step=8)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=128)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    batch_size = 1

    # --- Instantiate the model ---    
    model = PointCloudAutoencoder(latent_dim=latent_dim, hidden_dim=hidden_dim, lr=lr)
    model.net.latent_size = latent_dim  # careful, you may need to reinit layers here
    model.net.hidden_dim = hidden_dim
    optimizer = lambda params: torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    model.configure_optimizers = lambda: optimizer(model.net.parameters())

    # --- Load data ---
    train_loader, val_loader = utils.get_datasets(batch_size=batch_size)  # return PyG DataLoader objects

    # --- Callbacks ---
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    logger = CSVLogger(
            "optuna_logs",
            name=f"trial_{trial.number}",
            version=f"latent_{latent_dim}_hidden_{hidden_dim}"
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"optuna_checkpoints/trial_{trial.number}",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )
    
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        enable_progress_bar=True,
        callbacks=[
                early_stop,
                checkpoint_callback
                #PyTorchLightningPruningCallback(trial, monitor="val_loss")
                ],
        accelerator="auto",  # Automatically uses GPU if available
        devices="auto"
    )

    # --- Train ---
    trainer.fit(model, train_loader, val_loader)

    # --- Return final validation loss ---
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("Best trial:", study.best_trial.params)

# Visualize results
optuna.visualization.plot_optimization_history(study).show()

best_trial = study.best_trial
best_model_path = f"optuna_checkpoints/trial_{best_trial.number}/best_model.ckpt"
best_model = PointCloudAutoencoder.load_from_checkpoint(best_model_path)

loaders = utils.get_datasets()

utils.evaluate_model(best_model, loaders)

torch.cuda.empty_cache()
