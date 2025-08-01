from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir, model_nums):
        self.log_dir = Path(log_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.log_dir / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.run_dir))

        self.batch_weights = []
        self.model_names = [f"Model_{i}" for i in range(model_nums)]

    def record_batch_weights(self, weights):
        self.batch_weights.append(weights.detach().cpu().numpy())

    def log_epoch_weights(self, epoch):
        if not self.batch_weights:
            return

        all_weights = np.concatenate(self.batch_weights, axis=0)

        mean_weights = all_weights.mean(axis=0)

        heatmap_data = all_weights.T

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')

        ax.set_yticks(np.arange(len(self.model_names)))
        ax.set_yticklabels(self.model_names)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Model Index")
        ax.set_title(f"Model Weights Distribution - Epoch {epoch}")

        plt.colorbar(im, ax=ax, label="Weight Value")

        self.writer.add_figure("model_weights/heatmap", fig, epoch)

        for i, (name, weight) in enumerate(zip(self.model_names, mean_weights)):
            self.writer.add_scalar(f"model_weights/{name}", weight, epoch)

        self.batch_weights = []
        plt.close(fig)

    def close(self):
        self.writer.close()