import glob
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from tqdm import tqdm

from data_provider import data_loader
from data_provider.data_loader import TrainSet
from models.Ensemble import EnsembleModel
from models.SR import SymbolRegression
from utils.TrainingLogger import TrainingLogger
from config.config import PROJECT_ROOT, pool_root_path, device, epochs, batch_size, data_root_path, data_cache_dir


def train_ensemble_model(ensemble, dataloader, val_dataloader, epochs,
                         device, save_dir='checkpoints', log_dir='logs'):
    save_path = PROJECT_ROOT / save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = save_path / timestamp
    save_path.mkdir(exist_ok=True)

    log_dir = PROJECT_ROOT / log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    logger = TrainingLogger(log_dir, ensemble.num_models)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        ensemble.train()
        total_loss = 0
        total_samples = 0

        train_loop = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            ncols=100,
        )

        for x, y_true, prev_x, prev_y in train_loop:
            x, y_true, prev_x, prev_y = x.to(device), y_true.to(device), prev_x.to(device), prev_y.to(device)

            y_pred, selected_mask, weights, dist = ensemble.predict(x, prev_x, prev_y)
            logger.record_batch_weights(weights)

            reward = -torch.abs(y_pred - y_true).mean().item()

            batch_size = x.size(0)
            ensemble.weight_policy.rewards.extend([reward] * batch_size)

            loss = -reward
            total_loss += loss * batch_size
            total_samples += batch_size

            ensemble.update_policies(x, y_true, selected_mask)

        logger.log_epoch_weights(epoch)

        avg_train_loss = total_loss / total_samples
        train_losses.append(avg_train_loss)

        val_loss, val_accuracy = evaluate(ensemble, val_dataloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        ensemble.scheduler.step(val_loss)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': {
                    'select_policy': ensemble.select_policy.state_dict(),
                    'weight_policy': ensemble.weight_policy.state_dict(),
                    'performance_embedder': ensemble.performance_embedder.state_dict()
                },
                'optimizer_state_dict': ensemble.optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
            }, save_path / f'{timestamp}_best_model_summer.pth')

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': {
                    'select_policy': ensemble.select_policy.state_dict(),
                    'weight_policy': ensemble.weight_policy.state_dict(),
                    'performance_embedder': ensemble.performance_embedder.state_dict()
                },
                'optimizer_state_dict': ensemble.optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
            }, save_path / f'{timestamp}_checkpoint_epoch_{epoch + 1}_summer.pth')

    writer.close()
    logger.close()

    plot_training_curves(train_losses, val_losses, val_accuracies)

    return train_losses, val_losses, val_accuracies


def evaluate(system, dataloader, device='cpu'):
    system.eval()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y_true, prev_x, prev_y = batch
            x, y_true, prev_x, prev_y = x.to(device), y_true.to(device), prev_x.to(device), prev_y.to(device)

            y_pred, _, _, _ = system.predict(x, prev_x, prev_y)

            loss = torch.abs(y_pred - y_true).mean().item()
            total_loss += loss * x.size(0)

            accuracy = (torch.abs(y_pred - y_true) < 0.1).float().mean().item()
            total_accuracy += accuracy * x.size(0)

            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples

    return avg_loss, avg_accuracy


def plot_training_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = PROJECT_ROOT / 'results/plots'
    plt.tight_layout()
    plt.savefig(save_path / f'{timestamp}_training_curves.png')
    plt.close()


def load_all_datasets_by_name(root_path, cache_dir, val_filenames):
    files = glob.glob(os.path.join(PROJECT_ROOT, root_path, '*.csv'))
    filenames = [os.path.basename(f) for f in files]

    all_train_datasets = []
    all_val_datasets = []

    for name in filenames:
        data = data_loader.Dataset_Osaka_15min(root_path, name, cache_dir)
        dataset = TrainSet(data.train_X, data.train_Y, data.prev_X, data.prev_Y)

        if val_filenames and name in val_filenames:
            all_val_datasets.append(dataset)
        else:
            all_train_datasets.append(dataset)

    combined_train = ConcatDataset(all_train_datasets)
    combined_val = ConcatDataset(all_val_datasets)

    return combined_train, combined_val


def load_model_zoo(root_path):
    zoo = []
    files = glob.glob(os.path.join(PROJECT_ROOT, root_path, '*.txt'))
    filenames = [os.path.basename(f) for f in files]
    for name in filenames:
        sr = SymbolRegression(root_path, name)
        zoo.append(sr)
    return zoo


if __name__ == "__main__":
    print(f"Using device: {device}")

    train_dataset, val_dataset = load_all_datasets_by_name(data_root_path, data_cache_dir,
                                                           val_filenames=['A403.csv', 'A405.csv', 'A408.csv'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_zoo = load_model_zoo(pool_root_path)

    ensemble_model = EnsembleModel(model_zoo, device=device)
    ensemble_model.select_policy.to(device)
    ensemble_model.weight_policy.to(device)

    train_losses, val_losses, val_accuracies = train_ensemble_model(
        ensemble_model,
        train_dataloader,
        val_dataloader,
        epochs,
        device
    )
    print('Finished Training')
