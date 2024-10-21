import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from torchmetrics import Accuracy

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
    def prepare_data(self):
        # Download data if needed
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            train_size = int(0.2 * len(cifar_full))  # 20% for training
            val_size = len(cifar_full) - train_size   # 80% for validation
            self.train_data, self.val_data = random_split(
                cifar_full, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
        if stage == 'test' or stage is None:
            self.test_data = datasets.CIFAR10(
                self.data_dir, 
                train=False, 
                transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, persistent_workers=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=2*self.batch_size, persistent_workers=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=2*self.batch_size)

class CIFAR10Module(L.LightningModule):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Load ResNet18 model
        self.model = models.resnet18(weights=None)
        # Modify the last layer for CIFAR10 (10 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return [optimizer], [scheduler]

# Training script
def main():
    # Set seed
    L.seed_everything(42)

    # Initialize DataModule
    data_module = CIFAR10DataModule(batch_size=64)
    
    # Initialize model
    model = CIFAR10Module(learning_rate=0.001)
    
    # Configure logger
    logger = TensorBoardLogger("lightning_logs", name="cifar10_supervised_0.2")
    
    # Configure checkpoint callback
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath='checkpoints',
    #     filename='cifar10-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=3,
    #     mode='min',
    # )

    # Configure learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=30,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=[lr_monitor],
        deterministic=True,
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)

if __name__ == '__main__':
    main()