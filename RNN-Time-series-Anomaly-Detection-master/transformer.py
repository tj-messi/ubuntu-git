import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import shutil
from pathlib import Path
import numpy as np
import random

# 设置随机种子，确保结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

class TransformerPredictor(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout=0.5):
        super(TransformerPredictor, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        src = self.encoder(input)
        print("src shape:", src.shape)
        output = self.transformer_encoder(src)
        print("output shape after TransformerEncoder:", output.shape)
        output = self.decoder(output)
        return output

    def save_checkpoint(self, state, is_best):
        print("=> saving checkpoint..")
        args = state['args']
        checkpoint_dir = Path('save', args.data, 'checkpoint')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.filename).with_suffix('.pth')
        torch.save(state, checkpoint)
        if is_best:
            model_best_dir = Path('save', args.data,'model_best')
            model_best_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.filename).with_suffix('.pth'))
        print('=> checkpoint saved.')

    def load_checkpoint(self, args, checkpoint, input_size, output_size):
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size = args.prediction_window_size
        self.__init__(input_size=input_size, output_size=output_size,
                      d_model=args_.emsize, nhead=args_.nhead, num_layers=args_.nlayers,
                      dropout=args_.dropout)
        self.to(args.device)
        self.load_state_dict(checkpoint['state_dict'])
        return args_, start_epoch, best_val_loss

# 自定义数据集类
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, window_size):
        self.data = data
        self.target = target
        self.window_size = window_size

    def __getitem__(self, index):
        input_seq = self.data[index:index + self.window_size].unsqueeze(0)
        target = self.target[index + self.window_size].unsqueeze(0)
        return input_seq, target

    def __len__(self):
        return len(self.data) - self.window_size

# 数据预处理函数
def preprocess_data(data, train_ratio=0.8, window_size=10):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_dataset = TimeSeriesDataset(train_data, train_data, window_size)
    val_dataset = TimeSeriesDataset(val_data, val_data, window_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

# 训练函数
def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=100):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_val_loss,
            'args': {
                'data': 'test_data',
                'filename': 'transformer_model',
                'device': device,
                'emsize': model.encoder.in_features,
                'nhead': model.transformer_encoder.layers[0].nhead,
                'nlayers': len(model.transformer_encoder.layers),
                'dropout': model.transformer_encoder.layers[0].dropout,
                'epochs': num_epochs,
                'save_interval': 10,
                'prediction_window_size': window_size
            }
        }
        model.save_checkpoint(state, val_loss < best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

# 主函数
if __name__ == '__main__':
    # 生成一些随机的时间序列数据作为示例
    data = torch.randn(1000, 5)
    train_loader, val_loader = preprocess_data(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerPredictor(input_size=5, output_size=5, d_model=64, nhead=4, num_layers=3)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train(model, train_loader, val_loader, optimizer, criterion, device)