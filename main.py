import torch
from train import train_model
from model_CNN_test import KappaPredictorCNN
from model_UNet_test import UNet3D
import torch.nn as nn

if __name__ == '__main__':

    lr = 1e-3
    batch_size = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    #model = KappaPredictorCNN(in_channels=12).to(device)
    model = UNet3D(
        in_channels=12,  # Number of input variables in KappaDataset
        n_classes=2,    # kappa_SU and kappa_CA
        bilinear=True   # Use bilinear upsampling (smoother but potentially less sharp)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Run the training
    train_model('hamlite_sample_data_filtered.nc', epochs=1, plot=True, model=model, optimizer=optimizer, loss_fn=loss_fn, device = device, lr = lr, batch_size = batch_size, experiment_name="KappaPredictor")
