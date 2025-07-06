import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging 
import sys  
from pathlib import Path  
from typing import Callable, Tuple, Union 
from sklearn.neighbors import NearestNeighbors
import math
import cv2
sys.path.append("/homes/fhoerst/histo-projects/Dense/")  
import numpy as np  
import pandas as pd  
import torch 
import yaml 
from numba import njit  
from PIL import Image 
from scipy.ndimage import center_of_mass, distance_transform_edt 
from nuclei_segmentation.datasets.base_nuclei import nucleiDataset  
from nuclei_segmentation.utils.tools import fix_duplicates, get_bounding_box  
logger = logging.getLogger() 
logger.addHandler(logging.NullHandler()) 
from natsort import natsorted 
from typing import Tuple, List
import os
import matplotlib.pyplot as plt
import numpy as np

class PanNukeDataset(nucleiDataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        folds: Union[int, list[int]],
        transforms: Callable = None,
        stardist: bool = False,
        regression: bool = False,
        density: bool = False,
        cache_dataset: bool = False,
        is_train: bool = True,
    ) -> None:
        if isinstance(folds, int):
            folds = [folds]
    
        self.dataset = Path(dataset_path).resolve()  
        self.transforms = transforms  
        self.images = []  
        self.masks = [] 
        self.is_train=is_train
        
        self.img_names = []  
        self.folds = folds  
        self.cache_dataset = cache_dataset 
        self.stardist = stardist 
        self.regression = regression  
        self.density = density

        for fold in folds: 
            image_path = self.dataset / f"fold{fold}" / "images"  
            fold_images = [
                f for f in natsorted(image_path.glob("*.png")) if f.is_file()
            ]  
    
            for fold_image in fold_images: 
                mask_path = (
                    self.dataset / f"fold{fold}" / "masks" / f"{fold_image.stem}.npy"
                )  
                if mask_path.is_file():  
                    self.images.append(fold_image) 
                    self.masks.append(mask_path) 
                    self.img_names.append(fold_image.name) 
                else:
                    logger.debug(
                        "Found image {fold_image}, but no corresponding annotation file!"
                    ) 

        logger.info(f"Created Pannuke Dataset by using fold(s) {self.folds}") 
        logger.info(f"Resulting dataset length: {self.__len__()}") 
    
        if self.cache_dataset:
            self.cached_idx = []  
            self.cached_imgs = {}  
            self.cached_masks = {} 
            logger.info("Using cached dataset. Cache is built up during first epoch.") 
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:

        img_path = self.images[index]  
    
        if self.cache_dataset: 
            if index in self.cached_idx: 
                img = self.cached_imgs[index]  
                mask = self.cached_masks[index] 
            else:
                # cache file
                img = self.load_imgfile(index) 
                mask = self.load_maskfile(index) 
                self.cached_imgs[index] = img 
                self.cached_masks[index] = mask  
                self.cached_idx.append(index)  
    
        else:
            img = self.load_imgfile(index) 
            mask = self.load_maskfile(index)  
    
        if self.transforms is not None:  
            transformed = self.transforms(image=img, mask=mask)  
            img = transformed["image"]  
            mask = transformed["mask"]  
                
        gt_inst_map = mask[:, :, 0].copy().astype(np.int32) 
        gt_inst_map_only_fb = gt_inst_map.copy()
        gt_inst_map_only_fb[gt_inst_map_only_fb == -1] = 0  

        pse_inst_map = mask[:, :, 3].copy().astype(np.int32)  
        pse_inst_map_only_fb = pse_inst_map.copy()
        pse_inst_map_only_fb[pse_inst_map_only_fb == -1] = 0 


        den_map = mask[:, :, 1].copy()
        point_map = mask[:,:,2].copy().astype(np.int32)

        uncertain_mask_np = np.ones_like(pse_inst_map, dtype=np.int64)  
        uncertain_mask_np[pse_inst_map == -1] = 0 
        uncertain_mask_torch = torch.from_numpy(uncertain_mask_np).long().unsqueeze(0) 
        

        gt_np_map = mask[:, :, 0].copy() 
        gt_np_map[gt_np_map > 0] = 1  
        gt_np_map[gt_np_map == -1]=0
        gt_hv_map = PanNukeDataset.gen_instance_hv_map(gt_inst_map_only_fb) 

        pse_np_map = mask[:, :, 3].copy()  
        pse_np_map[pse_np_map > 0] = 1 
        pse_np_map[pse_np_map == -1]=0
        pse_hv_map = PanNukeDataset.gen_instance_hv_map(pse_inst_map_only_fb)  

        # torch convert
        img = torch.Tensor(img).type(torch.float32)  
        img = img.permute(2, 0, 1)  
        if torch.max(img) >= 5:  
            img = img / 255  

        mask_knn = self.generate_knn_mask(point_map, k_n=5, img_shape=(mask.shape[0], mask.shape[1]))

        masks = {
            "gt_instance_map": torch.Tensor(gt_inst_map).type(torch.int64), 
            "gt_nuclei_binary_map": torch.Tensor(gt_np_map).type(torch.int64),  
            "gt_hv_map": torch.Tensor(gt_hv_map).type(torch.float32),  
            "pse_instance_map": torch.Tensor(pse_inst_map).type(torch.int64), 
            "pse_nuclei_binary_map": torch.Tensor(pse_np_map).type(torch.int64),  
            "pse_hv_map": torch.Tensor(pse_hv_map).type(torch.float32), 
            "mask_knn": torch.Tensor(mask_knn).type(torch.int64),  
            "uncertain_mask": uncertain_mask_torch, 
        }  
    
        if self.density:  
            masks["den_map"] = torch.Tensor(den_map).type(torch.float32)  
            masks["point_map"]=torch.Tensor(point_map).type(torch.int64)

        if self.stardist:  
            dist_map = PanNukeDataset.gen_distance_prob_maps(gt_inst_map_only_fb) 
            stardist_map = PanNukeDataset.gen_stardist_maps(gt_inst_map_only_fb)  
            masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32) 
            masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)  
        if self.regression:  # 如果启用了回归
            masks["regression_map"] = PanNukeDataset.gen_regression_map(gt_inst_map_only_fb)  
    
        return img, masks, Path(img_path).name  # -------------------------------------------------------
    
    def __len__(self) -> int:
        return len(self.images)  # 返回数据集中的图像数量
    
    def set_transforms(self, transforms: Callable) -> None:
        self.transforms = transforms  # 更新图像变换
    
    def load_imgfile(self, index: int) -> np.ndarray:
        img_path = self.images[index]  # 获取图像路径
        return np.array(Image.open(img_path)).astype(np.uint8)  # 打开图像并转换为 NumPy 数组，类型为 uint8
    
    def load_maskfile(self, index: int) -> np.ndarray:

        mask_path = self.masks[index] 
        try:
            mask = np.load(mask_path, allow_pickle=True) 

            pse_inst_map = mask[()]["pse_inst_map"].astype(np.int32) 
            gt_inst_map = mask[()]["gt_inst_map"].astype(np.int32)
            den_map = mask[()]["den_map"].astype(np.float32)  
            point_map = mask[()]["point_map"].astype(np.int32) 

            mask = np.stack([gt_inst_map, den_map, point_map,pse_inst_map], axis=-1)# 堆叠为一个掩码数组
            return mask 
        except EOFError:
            print(f"{mask_path} error ")
            raise
    

    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:

        orig_inst_map = inst_map.copy()  
    
        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32) 
    
        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0) 
        for inst_id in inst_list:  
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)  
            inst_box = get_bounding_box(inst_map) 
            
            if inst_box[0] >= 2:
                inst_box[0] -= 2  
            if inst_box[2] >= 2:
                inst_box[2] -= 2 
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2  
            if inst_box[3] <= orig_inst_map.shape[1] - 2:
                inst_box[3] += 2 

      
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]  # 裁剪实例掩码到扩展后的边界框
    
            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue  
    
           
            inst_com = list(center_of_mass(inst_map)) 
    
            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5) 
    
            inst_x_range = np.arange(1, inst_map.shape[1] + 1) 
            inst_y_range = np.arange(1, inst_map.shape[0] + 1) 
            inst_x_range -= inst_com[1]  
            inst_y_range -= inst_com[0] 
    
            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)  
    
            # remove coord outside of instance
            inst_x[inst_map == 0] = 0 
            inst_y[inst_map == 0] = 0 
            inst_x = inst_x.astype("float32")  
            inst_y = inst_y.astype("float32")
    
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])  
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0]) 
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])  
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])  
    
            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]  
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]  
    
            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]  
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]  
    
        hv_map = np.stack([x_map, y_map])  
        return hv_map  
    

    @staticmethod
    def gen_distance_prob_maps(inst_map: np.ndarray) -> np.ndarray:
        inst_map = fix_duplicates(inst_map)  
        dist = np.zeros_like(inst_map, dtype=np.float64)  
        inst_list = list(np.unique(inst_map)) 
        if 0 in inst_list:
            inst_list.remove(0)  
    
        for inst_id in inst_list: 
            inst = np.array(inst_map == inst_id, np.uint8)  
    
            y1, y2, x1, x2 = get_bounding_box(inst) 
            y1 = y1 - 2 if y1 - 2 >= 0 else y1 
            x1 = x1 - 2 if x1 - 2 >= 0 else x1 
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2 
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2 
    
            inst = inst[y1:y2, x1:x2]  
    
            if inst.shape[0] < 2 or inst.shape[1] < 2:
                continue 
    
            inst_dist = distance_transform_edt(inst)  
            inst_dist = inst_dist.astype("float64")  
    
            max_value = np.amax(inst_dist) 
            if max_value <= 0:
                continue 
            inst_dist = inst_dist / (np.max(inst_dist) + 1e-10)  
    
            dist_map_box = dist[y1:y2, x1:x2]  
            dist_map_box[inst > 0] = inst_dist[inst > 0] 
    
        return dist 
    
    @staticmethod
    @njit
    def gen_stardist_maps(inst_map: np.ndarray) -> np.ndarray:
        n_rays = 32  
        dist = np.empty(inst_map.shape + (n_rays,), np.float32) 
    
        st_rays = np.float32((2 * np.pi) / n_rays)
        for i in range(inst_map.shape[0]):  
            for j in range(inst_map.shape[1]): 
                value = inst_map[i, j] 
                if value == 0:
                    dist[i, j] = 0  
                else:
                    for k in range(n_rays):  
                        phi = np.float32(k * st_rays)
                        dy = np.cos(phi) 
                        dx = np.sin(phi)
                        x, y = np.float32(0), np.float32(0) 
                        while True:
                            x += dx  
                            y += dy 
                            ii = int(round(i + x)) 
                            jj = int(round(j + y))  
                            if (
                                ii < 0
                                or ii >= inst_map.shape[0]
                                or jj < 0
                                or jj >= inst_map.shape[1]
                                or value != inst_map[ii, jj]
                            ):
                                t_corr = 1 - 0.5 / max(np.abs(dx), np.abs(dy))  
                                x -= t_corr * dx
                                y -= t_corr * dy
                                dst = np.sqrt(x**2 + y**2)  
                                dist[i, j, k] = dst 
                                break  
    
        return dist.transpose(2, 0, 1)  

    def KNN(self, points: List[Tuple[int, int]], k_n: int) -> List[List[float]]:

        if not points:
            return []
        n_samples = len(points)
        effective_k = min(k_n + 1, n_samples) if n_samples > 1 else 1
        if effective_k <= 1:
            return [[] for _ in points]
        nbrs = NearestNeighbors(n_neighbors=effective_k, algorithm='ball_tree').fit(points)
        distances, _ = nbrs.kneighbors(points)
        distances = distances[:, 1:effective_k] 
        return distances.tolist()

    def generate_knn_mask(
        self,
        point_map: np.ndarray,
        k_n: int = 3,
        img_shape: Tuple[int, int] = (256, 256),
        single_point_radius: int = 15
    ) -> np.ndarray:

        points = np.column_stack(np.where(point_map == 255))
        points = [(x, y) for y, x in points]

        mask_knn = np.zeros(img_shape, np.uint8)
        points_sum = len(points)

        if points_sum == 0:
            return mask_knn

        if points_sum == 1:
            single_point = points[0]
            center_x, center_y = int(round(single_point[0])), int(round(single_point[1]))
            radius = single_point_radius

            mask_knn = cv2.circle(mask_knn, (center_x, center_y), radius, 1, -1)

            mask_knn_float = mask_knn.astype(np.float32)
            mask_knn_float = cv2.GaussianBlur(mask_knn_float, (0, 0), sigmaX=radius/3, sigmaY=radius/3)
            mask_knn = (mask_knn_float > 0.5).astype(np.uint8)

            return mask_knn

        points_knn = self.KNN(points, k_n)
        for i, distances in enumerate(points_knn):
            if not distances:
                continue 

            current_k = min(k_n, len(distances))
            if current_k >= 1:
                r = distances[current_k - 1]  # 第 k 个最近邻的距离
                r_c = np.mean(distances[:current_k - 1]) if current_k > 1 else r
            else:
                r = 32 
                r_c = 32

            while current_k > 1 and r > r_c * 2:
                current_k -= 1
                r = distances[current_k - 1]
                r_c = np.mean(distances[:current_k - 1]) if current_k > 1 else r

            radius = max(math.floor(r), 10) 
            center_x, center_y = int(round(points[i][0])), int(round(points[i][1]))
            mask_knn = cv2.circle(mask_knn, (center_x, center_y), radius, 1, -1)

        for point in points:
            x, y = int(round(point[0])), int(round(point[1]))
            if mask_knn[y, x] == 0:
                mask_knn = cv2.circle(mask_knn, (x, y), 1, 1, -1)

        return mask_knn

    @staticmethod
    def gen_regression_map(inst_map: np.ndarray):
        n_directions = 2  
        dist = np.zeros(inst_map.shape + (n_directions,), np.float32).transpose(2, 0, 1)  # 初始化回归地图，形状为 (2, H, W)
        inst_map = fix_duplicates(inst_map) 
        inst_list = list(np.unique(inst_map)) 
        if 0 in inst_list:
            inst_list.remove(0)  
        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8) 
            y1, y2, x1, x2 = get_bounding_box(inst) 
            y1 = y1 - 2 if y1 - 2 >= 0 else y1  
            x1 = x1 - 2 if x1 - 2 >= 0 else x1  
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2 
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2  
    
            inst = inst[y1:y2, x1:x2]  
            y_mass, x_mass = center_of_mass(inst)  
            x_map = np.repeat(np.arange(1, x2 - x1 + 1)[None, :], y2 - y1, axis=0)
            y_map = np.repeat(np.arange(1, y2 - y1 + 1)[:, None], x2 - x1, axis=1) 
            x_dist_map = (x_map - x_mass) * np.clip(inst, 0, 1)  
            y_dist_map = (y_map - y_mass) * np.clip(inst, 0, 1) 
            dist[0, y1:y2, x1:x2] = x_dist_map 
            dist[1, y1:y2, x1:x2] = y_dist_map  
    
        return dist 
