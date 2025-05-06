import netCDF4 as nc
import numpy as np

file_path = r'C:\Users\radek\Documents\IT4I_projects\TerraDT\aerosol_ML_emulator\hamlite_sample_data.nc'
output_path = r'C:\Users\radek\Documents\IT4I_projects\TerraDT\aerosol_ML_emulator\hamlite_sample_data_filtered.nc'

# List of variables to keep
variables_to_keep = {
    "DU_lite", "SS_lite", "SU_lite", "CA_lite",
    "emi_DU", "emi_SS", "emi_BC", "emi_OC",
    "st", "apm1", "svo", "sd",
    "kappa_SU", "kappa_CA"
}

with nc.Dataset(file_path, 'r') as src, nc.Dataset(output_path, 'w') as dst:
    # Copy global attributes
    dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})

    # Determine dimensions needed for selected variables
    required_dims = set()
    for var_name in variables_to_keep:
        if var_name in src.variables:
            required_dims.update(src.variables[var_name].dimensions)
    
    # Copy required dimensions
    for dim_name in required_dims:
        dim = src.dimensions[dim_name]
        dst.createDimension(dim_name, (len(dim) if not dim.isunlimited() else None))

    # Copy variables
    for var_name in variables_to_keep:
        if var_name in src.variables:
            varin = src.variables[var_name]
            varout = dst.createVariable(var_name, varin.datatype, varin.dimensions)
            varout.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            varout[:] = varin[:]

    print("New file created with selected variables only:", output_path)
