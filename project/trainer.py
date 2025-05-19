import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ModelTrainer:
    """
    Encapsulates training loop for the DecoderOnlyTransformer.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataset: Dataset,
        batch_size: int = 32,
        lr: float = 3e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=1
        )

    def train(self, epochs: int = 5):
        self.model.train()
        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}")

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optim.zero_grad()
                logits = self.model(inputs)

                T, B, V = logits.shape
                loss = self.criterion(logits.view(T*B, V), labels.view(T*B))
                
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg = total_loss / len(self.loader)
            print(f"Epoch {epoch+1}/{epochs}  loss={avg:.4f}")
            
            # Update learning rate based on loss
            self.scheduler.step(avg_loss)
        
            # Save model if it's the best so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), "./models/decoder_only_model.pt")
                print(f"Model saved with loss: {best_loss:.4f}")
