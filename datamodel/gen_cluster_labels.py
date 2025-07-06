import os
import numpy as np
import cv2
import scipy.io as sio
from pathlib import Path
from skimage import measure, morphology, draw
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from shapely.geometry import Polygon
from tqdm import tqdm

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

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 5

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            norm = np.linalg.norm(t)
            if norm == 0:
                continue
            t /= norm
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        if vs.size == 0:
            continue
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def extract_points_from_npy(npy_file):
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        point_map = data["point_map"]
        non_zero_points = np.where(point_map != 0)
        points = np.array(list(zip(non_zero_points[1], non_zero_points[0])))
        return points
    except Exception as e:
        print(f"error in loading: {e}")
        return []

def create_voronoi_label(image_shape, points):
    h, w = image_shape

    if len(points) < 2:
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        return label_vor, False

    points_xy = points
    points_xy = np.unique(points_xy, axis=0)
    if points_xy.shape[0] < 2:
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        return label_vor, False

    try:
        vor = Voronoi(points_xy)
    except Exception as e:
        print(f"Voronoi computation failed for points: {points_xy} with error: {e}")
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        return label_vor, False

    try:
        regions, vertices = voronoi_finite_polygons_2d(vor, radius=max(h, w) * 5)
    except Exception as e:
        print(f"Voronoi finite polygons failed with error: {e}")
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        return label_vor, False

    box = Polygon([(0, 0), (0, h), (w, h), (w, 0)])
    edges = np.zeros((h, w), dtype=bool)
    region_masks = np.zeros((h, w), dtype=np.int16)
    count = 1

    for region in regions:
        if len(region) == 0:
            continue
        polygon_xy = vertices[region]
        poly = Polygon(polygon_xy).intersection(box)

        if poly.is_empty or poly.exterior is None:
            continue
        poly_coords = np.array(poly.exterior.coords)
        if poly_coords.shape[0] < 3:
            continue

        mask = poly2mask(poly_coords[:, 1], poly_coords[:, 0], (h, w))

        edge = mask & (~morphology.erosion(mask, morphology.disk(1)))
        edges |= edge
        region_masks[mask] = count
        count += 1

    label_point = np.zeros((h, w), dtype=np.uint8)
    for p in points:
        x, y = p
        if 0 <= y < h and 0 <= x < w:
            label_point[y, x] = 255
    label_point_dilated = morphology.dilation(label_point, morphology.disk(2))

    label_vor = np.zeros((h, w, 3), dtype=np.uint8)
    closed_edges = morphology.closing(edges, morphology.disk(1))
    label_vor[:, :, 0] = closed_edges.astype(np.uint8) * 255
    label_vor[:, :, 1] = (label_point_dilated > 0).astype(np.uint8) * 255

    return label_vor, True

def create_cluster_label(original_img, points, label_vor):
    h, w, _ = original_img.shape

    label_point = np.zeros((h, w), dtype=np.uint8)
    for p in points:
        x, y = p
        if 0 <= y < h and 0 <= x < w:
            label_point[y, x] = 255

    dist_embeddings = distance_transform_edt(255 - label_point).reshape(-1, 1)
    clip_dist_embeddings = np.clip(dist_embeddings, 0, 20)

    color_embeddings = original_img.reshape(-1, 3).astype(float) / 10
    embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)

    if embeddings.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(embeddings)
    clusters = np.reshape(kmeans.labels_, (h, w))

    overlap_nums = [np.sum((clusters == i) * (label_point > 0)) for i in range(3)]
    nuclei_idx = np.argmax(overlap_nums) if np.any(overlap_nums) else 0

    remain_indices = [x for x in range(3) if x != nuclei_idx]
    dilated_label_point = morphology.binary_dilation(label_point, morphology.disk(5))
    if len(remain_indices) == 0:
        background_idx = (nuclei_idx + 1) % 3
    else:
        overlap_nums_bg = [np.sum((clusters == i) * dilated_label_point) for i in remain_indices]
        background_idx = remain_indices[np.argmin(overlap_nums_bg)]

    nuclei_cluster = (clusters == nuclei_idx)
    background_cluster = (clusters == background_idx)

    nuclei_labeled = measure.label(nuclei_cluster)
    initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 30)
    refined_nuclei = np.zeros((h, w), dtype=bool)

    if np.any(label_vor[:, :, 0] == 0):
        voronoi_nucleis = measure.label(label_vor[:, :, 0] == 0)
        voronoi_nucleis = morphology.dilation(voronoi_nucleis, morphology.disk(2))
    else:
        voronoi_nucleis = np.ones((h, w), dtype=int)

    nuclei_indices = np.unique(voronoi_nucleis)
    nuclei_indices = nuclei_indices[nuclei_indices != 0]

    for ci in nuclei_indices:
        nuclei_i = (voronoi_nucleis == ci)
        nucleus_i = nuclei_i & initial_nuclei

        if np.any(nucleus_i):
            nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
            nucleus_i_dilated_filled = binary_fill_holes(nucleus_i_dilated)
            nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
            refined_nuclei |= nucleus_i_final

    for p in points:
        x, y = p
        if 0 <= y < h and 0 <= x < w:
            if not refined_nuclei[y, x]:
                rr, cc = draw.disk((y, x), radius=10, shape=refined_nuclei.shape)
                refined_nuclei[rr, cc] = True

    boundary_mask = label_vor[:, :, 0] > 0
    nuclei_mask_no_boundary = refined_nuclei & (~boundary_mask)

    inst_map, num_instances = measure.label(nuclei_mask_no_boundary, connectivity=2, return_num=True)
    uncertain_mask = np.zeros((h, w), dtype=bool)

    for instance_id in range(1, num_instances + 1):
        instance_mask = inst_map == instance_id
        points_in_instance = np.any(label_point[instance_mask] == 255)
        if not points_in_instance:
            uncertain_mask |= instance_mask

    inst_map[uncertain_mask] = -1

    refined_label = np.zeros((h, w, 3), dtype=np.uint8)
    label_point_dilated = morphology.dilation(label_point, morphology.disk(10))
    refined_label[:, :, 0] = ((background_cluster) & (~refined_nuclei) & (label_point_dilated == 0)).astype(np.uint8) * 255
    refined_label[:, :, 1] = (refined_nuclei).astype(np.uint8) * 255

    return refined_label

def label2rgb(label, bg_label=0):
    unique_labels = np.unique(label)
    unique_labels = unique_labels[unique_labels != bg_label]
    color_map = {label: np.random.randint(0, 255, 3) for label in unique_labels}
    color_map[bg_label] = [0, 0, 0]

    h, w = label.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, color in color_map.items():
        rgb_image[label == lbl] = color

    return rgb_image

def create_instance_map(label_vor, refined_label):
    background_mask = refined_label[:, :, 0] > 0
    nuclei_mask = refined_label[:, :, 1] > 0
    uncertain_mask = (refined_label[:, :, 0] == 0) & (refined_label[:, :, 1] == 0)

    boundary_mask = label_vor[:, :, 0] > 0
    nuclei_mask_no_boundary = nuclei_mask & (~boundary_mask)

    inst_map, num_instances = measure.label(nuclei_mask_no_boundary, connectivity=2, return_num=True)

    label_point = (label_vor[:, :, 1] > 0).astype(np.uint8)
    for instance_id in range(1, num_instances + 1):
        instance_mask = inst_map == instance_id
        points_in_instance = np.any(label_point[instance_mask] == 1)
        if not points_in_instance:
            uncertain_mask |= instance_mask

    inst_map[uncertain_mask] = -1
    return inst_map

def process_fold(fold):
    base_dir = Path("/path/dataset")
    fold_dir = base_dir / f"fold{fold}"
    point_dir = fold_dir / "masks"
    img_dir = fold_dir / "images"
    voronoi_save_dir = fold_dir / "voronoi_labels"
    cluster_save_dir = fold_dir / "cluster_labels"
    instance_save_dir = fold_dir / "label_instance"
    create_folder(voronoi_save_dir)
    create_folder(cluster_save_dir)
    create_folder(instance_save_dir)

    img_files = sorted(img_dir.glob("*.png"))
    for idx, img_path in enumerate(tqdm(img_files, desc=f"Processing Fold {fold}")):
        name = img_path.stem
        npy_file = point_dir / f"{name}.npy"

        vor_path = voronoi_save_dir / f"{name}.png"
        clu_path = cluster_save_dir / f"{name}.png"
        inst_path = instance_save_dir / f"{name}.npy"

        if vor_path.exists() and clu_path.exists() and inst_path.exists():
            continue

        if not npy_file.exists():
            print(f"Warning: Point label file '{npy_file}' does not exist. Skipping.")
            continue

        original_img = cv2.imread(str(img_path))
        if original_img is None:
            print(f"Warning: Unable to read image '{img_path}'. Skipping.")
            continue
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        try:
            points = extract_points_from_npy(npy_file)
        except KeyError as e:
            print(f"Error: {e}. Skipping '{npy_file}'.")
            continue

        label_vor, vor_created = create_voronoi_label((original_img.shape[0], original_img.shape[1]), points)
        cv2.imwrite(str(vor_path), cv2.cvtColor(label_vor, cv2.COLOR_RGB2BGR))

        refined_label = create_cluster_label(original_img, points, label_vor)
        cv2.imwrite(str(clu_path), cv2.cvtColor(refined_label, cv2.COLOR_RGB2BGR))

        inst_map = create_instance_map(label_vor, refined_label)
        inst_dict = {"inst_map": inst_map}
        np.save(str(inst_path), inst_dict)



def main():
    folds = [0,]
    for f in folds:
        process_fold(f)

if __name__ == '__main__':
    main()