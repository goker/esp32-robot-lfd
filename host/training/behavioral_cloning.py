"""
Behavioral Cloning Policy

Simple but effective baseline for learning from demonstrations.
Maps observations directly to actions using supervised learning.

Usage:
    python -m host.training.behavioral_cloning \
        --dataset datasets/pick_place \
        --epochs 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import argparse
from tqdm import tqdm


@dataclass
class BCConfig:
    """Behavioral Cloning configuration."""
    image_size: Tuple[int, int] = (240, 320)
    image_channels: int = 3
    state_dim: int = 5
    action_dim: int = 4
    hidden_dims: Tuple[int, ...] = (256, 256)
    use_image: bool = True
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    augment: bool = True
    color_jitter: float = 0.1
    random_crop: bool = True


class ImageEncoder(nn.Module):
    """Simple CNN encoder for visual observations."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 240, 320)
            flat_size = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        if x.max() > 1.0:
            x = x.float() / 255.0
        features = self.conv(x)
        return self.fc(features)


class BCPolicy(nn.Module):
    """Behavioral Cloning policy network."""

    def __init__(self, config: BCConfig):
        super().__init__()
        self.config = config
        if config.use_image:
            self.image_encoder = ImageEncoder(output_dim=256)
            input_dim = 256 + config.state_dim
        else:
            self.image_encoder = None
            input_dim = config.state_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, config.action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = []
        if self.image_encoder is not None and image is not None:
            img_features = self.image_encoder(image)
            features.append(img_features)
        if state is not None:
            features.append(state)
        if not features:
            raise ValueError("At least one of image or state must be provided")
        x = torch.cat(features, dim=-1)
        action = self.mlp(x)
        return action

    def predict(
        self,
        image: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        device: str = "cpu"
    ) -> np.ndarray:
        """Predict action for deployment."""
        self.eval()
        with torch.no_grad():
            if image is not None:
                image = torch.from_numpy(image).unsqueeze(0).to(device)
            if state is not None:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = self.forward(image, state)
        return action.cpu().numpy().squeeze()


class RobotDataset(Dataset):
    """Dataset loader for recorded demonstrations."""

    def __init__(self, data_dir: str, use_image: bool = True, transform=None):
        self.data_dir = Path(data_dir)
        self.use_image = use_image
        self.transform = transform
        self.episode_files = sorted(self.data_dir.glob("episode_*.hdf5"))
        if not self.episode_files:
            raise ValueError(f"No episodes found in {data_dir}")
        self.index = []
        for ep_file in self.episode_files:
            with h5py.File(ep_file, 'r') as f:
                num_steps = f.attrs['num_steps']
                for t in range(num_steps - 1):
                    self.index.append((ep_file, t))
        print(f"Loaded {len(self.episode_files)} episodes, {len(self.index)} timesteps")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_file, t = self.index[idx]
        with h5py.File(ep_file, 'r') as f:
            if self.use_image:
                image = f['observations/images'][t]
            else:
                image = None
            joint_pos = f['observations/joint_positions'][t]
            gripper = f['observations/gripper_state'][t]
            action = f['actions/joint_targets'][t + 1]
        state = np.concatenate([joint_pos, [gripper]]).astype(np.float32)
        sample = {'state': state, 'action': action.astype(np.float32)}
        if image is not None:
            if self.transform:
                image = self.transform(image)
            sample['image'] = image.astype(np.float32)
        return sample


def train_bc(
    config: BCConfig,
    dataset_path: str,
    output_dir: str = "outputs",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train behavioral cloning policy."""
    print(f"\n{'='*50}")
    print("Training Behavioral Cloning Policy")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = RobotDataset(dataset_path, use_image=config.use_image)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    policy = BCPolicy(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_loss = float('inf')
    for epoch in range(config.epochs):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch in pbar:
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            image = batch.get('image')
            if image is not None:
                image = image.to(device)

            pred_action = policy(image=image, state=state)
            loss = F.mse_loss(pred_action, action)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss, 'config': config
            }, output_path / "bc_policy_best.pt")
            print(f"  Saved best model (loss={best_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': policy.state_dict(),
                'loss': avg_loss, 'config': config
            }, output_path / f"bc_policy_epoch{epoch+1}.pt")

    torch.save({
        'epoch': config.epochs, 'model_state_dict': policy.state_dict(), 'config': config
    }, output_path / "bc_policy_final.pt")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning policy")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    config = BCConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, use_image=not args.no_image
    )
    train_bc(config, args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
