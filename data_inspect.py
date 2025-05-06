import netCDF4 as nc
import xarray as xr

# Open the NetCDF file
# file_path = r'C:\Users\radek\Documents\IT4I_projects\TerraDT\aerosol_ML_emulator\hamlite_sample_data.nc'
file_path = r'C:\Users\radek\Documents\IT4I_projects\TerraDT\aerosol_ML_emulator\hamlite_sample_data_filtered.nc'

# Using netCDF4
with nc.Dataset(file_path, 'r') as dataset:
    print("NetCDF4 Dataset Information:")
    print("Dimensions:", list(dataset.dimensions.keys()))
    print("Variables:", list(dataset.variables.keys()))
    
    # Print some details about the variables
    for var_name in dataset.variables:
        var = dataset.variables[var_name]
        print(f"\nVariable: {var_name}")
        print(f"Dimensions: {var.dimensions}")
        print(f"Shape: {var.shape}")
        print(f"Datatype: {var.dtype}")

# Using xarray (provides more convenient data manipulation)
ds = xr.open_dataset(file_path)
print("\nXarray Dataset Summary:")
print(ds)

######################################################333

