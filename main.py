import torch
from train_time_approach import train_model
#from train import train_model
#from model_CNN_test import KappaPredictorCNN
#from model_UNet_test import UNet3D
from model_ConvLSTM import KappaPredictorConvLSTM
from model_UNet3D_temporal import UNet3DTemporal
import torch.nn as nn
from utils import visualize_mlflow_prediction

if __name__ == '__main__':

    lr = 1e-4
    batch_size = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    #model = KappaPredictorCNN(in_channels=12).to(device)
    
    #model = UNet3D(
    #    in_channels=12,  # Number of input variables in KappaDataset
    #    n_classes=2,    # kappa_SU and kappa_CA
    #    bilinear=True   # Use bilinear upsampling (smoother but potentially less sharp)
    #).to(device)

    # Option 1: ConvLSTM baseline
    # model = KappaPredictorConvLSTM(
    #     in_channels=12,
    #     hidden_channels=32,
    #     num_layers=1,
    #     kernel_size=3,
    #     out_channels=2
    # ).to(device)

    # Option 2: Memory-efficient 3D U-Net with temporal fusion
    model = UNet3DTemporal(
        in_channels=12,
        out_channels=2,
        base_ch=32,     # reduce/increase for memory vs capacity
        depth=3,        # 2 or 3 recommended for large volumes
        temporal_mode='attn',  # 'last' | 'mean' | 'attn'
        attn_heads=4,
        use_residual=True,      # enable residual blocks
        use_se=True,            # enable SE channel attention
        fuse_multiscale=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Run the training
    train_model('hamlite_sample_data_filtered.nc', epochs=5, plot=True, model=model, 
        optimizer=optimizer, loss_fn=loss_fn, device = device, n_splits = 1,
        lr = lr, batch_size = batch_size, experiment_name="KappaPredictor", 
        seq_len=3, show_fold_plot=False)
    
    #visualize_mlflow_prediction(
    #    run_id="7250ffe88ee943dfa03be8e347b91860",
    #    nc_path="hamlite_sample_data_filtered.nc",
    #    lev_indices=[0, 10],  # levels to visualize
    #    model_name="best_model"
    #)
    
