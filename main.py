import torch
from train import train_model
from model_CNN_test import KappaPredictorCNN
import torch.nn as nn

if __name__ == '__main__':

    lr = 1e-3
    batch_size = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = KappaPredictorCNN(in_channels=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Run the training
    train_model('hamlite_sample_data_filtered.nc', epochs=1, plot=True, model=model, optimizer=optimizer, loss_fn=loss_fn, device = device, lr = lr, batch_size = batch_size)
