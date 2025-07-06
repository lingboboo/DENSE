import os
import numpy as np
import matplotlib.pyplot as plt

def process_and_visualize_patches(input_dir_cluseg, input_dir_gtseg, output_base_dir, fold_dirs):
    for fold in fold_dirs:
        cluseg_dir = os.path.join(input_dir_cluseg, fold, "")
        gtseg_dir = os.path.join(input_dir_gtseg, fold, "")
        point_dir = os.path.join(input_dir_gtseg, fold, "")
        den_dir = os.path.join(input_dir_gtseg, fold, "")
        output_fold_dir = os.path.join(output_base_dir, fold, "")
        os.makedirs(output_fold_dir, exist_ok=True)
        
        cluseg_files = sorted([f for f in os.listdir(cluseg_dir) if f.endswith('.npy')])
        gtseg_files = sorted([f for f in os.listdir(gtseg_dir) if f.endswith('.npy')])
        point_files = sorted([f for f in os.listdir(point_dir) if f.endswith('.npy')])
        den_files = sorted([f for f in os.listdir(den_dir) if f.endswith('.npy')])
        
        assert cluseg_files == gtseg_files, f"Mismatch in files between {cluseg_dir} and {gtseg_dir}"

        for idx, file_name in enumerate(cluseg_files):
            cluseg_path = os.path.join(cluseg_dir, file_name)
            gtseg_path = os.path.join(gtseg_dir, file_name)
            point_path = os.path.join(point_dir, file_name)
            den_path = os.path.join(den_dir, file_name)

            cluseg_data = np.load(cluseg_path, allow_pickle=True).item()
            gtseg_data = np.load(gtseg_path, allow_pickle=True).item()
            point_data = np.load(point_path, allow_pickle=True).item()
            den_data = np.load(den_path, allow_pickle=True).item()
            
            combined_data = {
                "gt_inst_map": gtseg_data["gt_inst_map"],
                "den_map": den_data["density_map"],
                "point_map": point_data["point_map"],
                "pse_inst_map": cluseg_data["pse_inst_map"]
            }
            
            save_path = os.path.join(output_fold_dir, file_name)
            np.save(save_path, combined_data)


input_dir_cluseg = "/path/dataset/"
input_dir_gtseg = "/path/dataset/"
output_base_dir = "/path/dataset/"
fold_dirs = ["fold0",]

process_and_visualize_patches(
    input_dir_cluseg,
    input_dir_gtseg,
    output_base_dir,
    fold_dirs
)