# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math

class SimpleViTSequence(nn.Module):
    """ViT simplifié et optimisé pour la reconnaissance de séquences de chiffres"""
    def __init__(self, config, device):
        super().__init__()

        img_size = 120
        patch_size = config['w_width']  # 10
        embed_dim = config['hidden_size']  # 128
        num_heads = config['num_heads']  # 4
        depth = config['num_layers']  # 2

        self.seq_len = 5
        self.output_dim = config['num_classes']
        self.DEVICE = device

        # Patch embedding simplifié
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2  # 144

        # Positional encoding (sans CLS token pour simplifier)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(config['dropout'])

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Tête de classification
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(embed_dim, self.seq_len * self.output_dim)
        )

        # Initialisation
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.size(0)

        # Patch embedding: B x 1 x 120 x 120 -> B x 128 x 12 x 12
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # B x 144 x 128

        # Ajouter positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)  # B x 144 x 128

        # Global average pooling
        x = x.transpose(1, 2)  # B x 128 x 144
        x = self.pool(x).squeeze(-1)  # B x 128

        # Classification
        x = self.head(x)  # B x 50
        x = x.view(B, self.seq_len, self.output_dim)  # B x 5 x 10

        return x


def sequence_loss(logits, targets, pad_idx=14):
    """
    Loss function pour la prédiction de séquences.

    Args:
        logits: (B, seq_len, num_classes) - prédictions du modèle
        targets: (B, seq_len) - ground truth
        pad_idx: valeur de padding à ignorer

    Returns:
        loss: scalar
    """
    B, seq_len, num_classes = logits.shape

    # Reshape pour CrossEntropyLoss
    logits_flat = logits.reshape(-1, num_classes)
    targets_flat = targets.reshape(-1)

    # CrossEntropyLoss
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_idx, reduction='mean')

    return loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Learning rate scheduler avec warmup puis cosine decay"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_loop(dataloader, model, loss_fn, optimizer, scheduler=None):
    """Boucle d'entraînement optimisée"""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        X = batch[0].to(model.DEVICE)
        y = batch[1].to(model.DEVICE)
        # Correction: indices 1 to 6 for the 5 digits (0 is START)
        targets = y[:, 1:6]  

        # Forward pass
        logits = model(X.float())
        loss = loss_fn(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Metrics
        epoch_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def valid_loop(dataloader, model, loss_fn):
    """Boucle de validation"""
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(model.DEVICE)
            y = batch[1].to(model.DEVICE)
            # Correction: indices 1 to 6 for the 5 digits
            targets = y[:, 1:6]

            # Forward pass
            logits = model(X.float())
            loss = loss_fn(logits, targets)

            # Metrics
            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()

            # Sequence accuracy (tous les 5 chiffres corrects)
            correct_sequences += (preds == targets).all(dim=1).sum().item()
            total_sequences += targets.size(0)

    avg_loss = epoch_loss / len(dataloader)
    digit_accuracy = correct / total if total > 0 else 0
    seq_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0

    return avg_loss, digit_accuracy, seq_accuracy


def compute_accuracy(dataloader, model):
    """Calcule l'accuracy détaillée"""
    model.eval()
    correct_digits = 0
    correct_sequences = 0
    total_digits = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(model.DEVICE)
            y = batch[1].to(model.DEVICE)
            # Correction: indices 1 to 6 for the 5 digits
            targets = y[:, 1:6]

            logits = model(X.float())
            preds = logits.argmax(dim=-1)

            # Digit accuracy
            correct_digits += (preds == targets).sum().item()
            total_digits += targets.numel()

            # Sequence accuracy
            correct_sequences += (preds == targets).all(dim=1).sum().item()
            total_sequences += targets.size(0)

    digit_acc = correct_digits / total_digits if total_digits > 0 else 0
    seq_acc = correct_sequences / total_sequences if total_sequences > 0 else 0

    return digit_acc, seq_acc
