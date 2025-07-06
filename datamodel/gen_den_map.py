import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

def remap_label(pred, by_size=False):
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def extract_points_from_npy(npy_file):
    point_map = np.load(npy_file)
    points = np.column_stack(np.where(point_map == 255))
    return points.tolist()

def generate_density_map(points, image_shape=(512, 512)):

    """
    Generates a density map from a list of points. 
    It calculates the major and minor axes of ellipses centered at each point, using PCA to determine the orientation. 
    The density map is created by placing Gaussian ellipses at each point.
    """
    if not points:
        return np.zeros(image_shape, dtype=np.float32)
    
    points_array = np.array(points)
    num_points = len(points_array)

    major_axis_range = (35, 50)

    k_neighbors = 3 # choose

    if num_points <= 1:
        angles = [0.0]
        major_axes = [30]
        minor_axes = [22]
    else:
        actual_k = min(k_neighbors, num_points - 1)
        nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='ball_tree').fit(points_array)
        distances, indices = nbrs.kneighbors(points_array)
        angles = []
        for i, point in enumerate(points_array):
            neighbor_idxs = indices[i][1:actual_k + 1]
            neighbors = points_array[neighbor_idxs]
            neighbors_centered = neighbors - point
            if len(neighbors_centered) < 2:
                angle = 0.0
            else:
                pca = PCA(n_components=2)
                pca.fit(neighbors_centered)
                principal_component = pca.components_[0]
                angle = np.arctan2(principal_component[1], principal_component[0]) * (180 / np.pi)
            angles.append(angle)
        angles = np.array(angles)

        max_distances = distances[:, 1:k_neighbors + 1].mean(axis=1)

        major_axes = np.interp(max_distances, (max_distances.min(), max_distances.max()), major_axis_range)
        minor_axes = major_axes * 0.75

        real_neighbors_count = np.sum((distances[:, 1:k_neighbors + 1] > 0), axis=1)
        if actual_k == 1:
            min_neighbor_dist = np.min(distances[:, 1:actual_k + 1], axis=1)
            major_axes = np.minimum(min_neighbor_dist / 2, 40)
            minor_axes = major_axes * 0.75
        elif actual_k == 2:
            two_nearest_mean = (distances[:, 1] + distances[:, 2]) / 2.0
            condition_indices = (real_neighbors_count == 2)
            major_axes[condition_indices] = two_nearest_mean[condition_indices]
            minor_axes = major_axes * 0.75
        elif actual_k >= 3:
            first_three_distances = distances[:, 1:4]
            max_first_three_distances = first_three_distances.max(axis=1)
            condition = np.all(first_three_distances < major_axes[:, np.newaxis], axis=1)
            major_axes = np.where(condition, max_first_three_distances, major_axes)
            minor_axes = major_axes * 0.75

    density_map = np.zeros(image_shape, dtype=np.float32)
    height, width = image_shape

    for i, point in enumerate(points_array):
        y_center, x_center = point[0], point[1]
        major_axis_length = int(major_axes[i])
        minor_axis_length = int(minor_axes[i])
        angle = angles[i]

        y_grid, x_grid = np.ogrid[:height, :width]
        X = x_grid - x_center
        Y = y_grid - y_center

        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        X_rot = X * cos_t + Y * sin_t
        Y_rot = -X * sin_t + Y * cos_t

        a = major_axis_length / 2.0
        b = minor_axis_length / 2.0

        sigma_x = a / 2.0
        sigma_y = b / 2.0

        gauss = np.exp(-(0.5 * (X_rot ** 2) / (sigma_x ** 2) + 0.5 * (Y_rot ** 2) / (sigma_y ** 2)))

        ellipse_mask = (X_rot ** 2 / a ** 2 + Y_rot ** 2 / b ** 2) <= 1
        gauss[~ellipse_mask] = 0

        density_map[ellipse_mask] = np.maximum(density_map[ellipse_mask], gauss[ellipse_mask])

    return density_map

def extract_stem(filename):
    return Path(filename).stem

def process_fold(fold, input_path, output_path, point_dir, visualize=True, visualize_samples=5) -> None:
    """
    Processes a specific fold of the dataset. It reads images and corresponding point maps, 
    generates density maps, and saves the results. 
    Optionally, it visualizes a few samples of the original images and their corresponding density maps.
    """
    fold_input_path = Path(input_path) / f"fold{fold}" / "images"
    fold_output_path = Path(output_path) / f"fold{fold}"
    point_npy_path = Path(point_dir) / f"fold{fold}" / "point_map"
    ell_labels_path = fold_output_path / "cpm_den_choice"
    ell_labels_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing Fold: {fold}")

    image_files = sorted(fold_input_path.glob("*.png"))
    num_images = len(image_files)
    print(f"find {num_images} images")

    for idx, image_path in enumerate(tqdm(image_files, desc=f"Fold {fold}")):
        image_name = image_path.stem
        npy_filename = f"{image_name}.npy"
        npy_path = point_npy_path / npy_filename

        if not npy_path.exists():
            print(f" {npy_path} not exist")
            continue

        image = np.array(Image.open(image_path).convert('L'))
        image_shape = image.shape

        try:
            points = extract_points_from_npy(npy_path)
        except KeyError as e:
            print(f"error: {e}")
            continue

        density_map = generate_density_map(points, image_shape=image.shape)

        outdict = {"density_map": density_map}
        out_label_path = ell_labels_path / f"{image_name}.npy"
        np.save(out_label_path, outdict)

        if visualize and idx < visualize_samples:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(image, cmap='gray')
            axs[0].set_title(f'Original Image Fold{fold}_{image_name}')
            axs[0].axis('off')

            axs[1].imshow(density_map, cmap='jet')
            axs[1].set_title(f'Density Map Fold{fold}_{image_name}')
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

input_path = Path("/path/dataset")
output_path = Path("/path/dataset_for_train")
point_dir = Path("/path/dataset")

for fold in [0,]:
    process_fold(fold, input_path, output_path, point_dir, visualize=True, visualize_samples=5)