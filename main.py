import torch
from train_time_approach import train_model
#from train import train_model
from model_CNN_test import KappaPredictorCNN
from model_UNet_test import UNet3D
import torch.nn as nn
from utils import visualize_mlflow_prediction

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
    train_model('hamlite_sample_data_filtered.nc', epochs=25, plot=True, model=model, 
        optimizer=optimizer, loss_fn=loss_fn, device = device, n_splits = 3,
        lr = lr, batch_size = batch_size, experiment_name="KappaPredictor")
    
    #visualize_mlflow_prediction(
    #    #run_id="0b6d8924840d4773bb0e32330acf1327",
    #    run_id="111afb2fc5ba4bc791285532389e6ede",
    #    nc_path="hamlite_sample_data_filtered.nc",
    #    lev_indices=[0, 10],  # levels to visualize
    #    model_name="fold0/best_model"
    #)
    
