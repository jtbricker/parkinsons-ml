from pml.data_access import DataFile
from pml.data import DiffusionData

if __name__ == "__main__":
    diffusion_data_file = DataFile("diffusion mri data", 'data/all_data.xlsx')
    diffusion_data_file.extract_or_create()

    data = DiffusionData(diffusion_data_file.df, {'classes':{1:0,2:1}})
    print(data.data)