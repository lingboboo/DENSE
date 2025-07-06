import warnings
from typing import Tuple, Literal

import cv2
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
import torch

from .tools import get_bounding_box, remove_small_objects

def noop(*args, **kargs):
    pass

warnings.warn = noop

class DetectionnucleiPostProcessor:
    def __init__(self, magnification: Literal[20, 40] = 40, gt: bool = False) -> None:
        self.magnification = magnification
        self.gt = gt

        if magnification == 40:
            self.object_size = 10
            self.k_size = 21
        elif magnification == 20:
            self.object_size = 3
            self.k_size = 11
        else:
            raise NotImplementedError("Unknown magnification")
        if gt:
            self.object_size = 100
            self.k_size = 21

    def post_process_nuclei_segmentation(self, pred_map: np.ndarray) -> Tuple[np.ndarray, dict]:
        num_channels = pred_map.shape[-1]
        
        if num_channels == 4:
            nuclei_binary_map = pred_map[..., 1:2]
            hv_map = pred_map[..., 2:4]
            proc_map = np.concatenate([nuclei_binary_map, hv_map], axis=-1)
        elif num_channels == 3:
            proc_map = pred_map
        else:
            raise ValueError(f"Unexpected number of channels in pred_map: {num_channels}")

        proced_pred = self.__proc_np_hv(proc_map, object_size=self.object_size, ksize=self.k_size)

        inst_id_list = np.unique(proced_pred)
        inst_id_list = inst_id_list[inst_id_list != 0]
        inst_info_dict = {}

        for inst_id in inst_id_list:
            inst_map_single = proced_pred == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map_single)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_crop = inst_map_single[rmin:rmax, cmin:cmax].astype(np.uint8)
            contours, _ = cv2.findContours(inst_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = contours[0].squeeze()
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            moments = cv2.moments(inst_crop)
            if moments["m00"] == 0:
                continue
            centroid = [moments["m10"] / moments["m00"] + cmin, moments["m01"] / moments["m00"] + rmin]
            centroid = np.array(centroid)
            contour[:, 0] += cmin
            contour[:, 1] += rmin
            inst_info_dict[inst_id] = {
                "bbox": inst_bbox,
                "centroid": centroid,
                "contour": contour,
            }

        return proced_pred, inst_info_dict

    def __proc_np_hv(self, pred: np.ndarray, object_size: int = 10, ksize: int = 21) -> np.ndarray:
        pred = np.array(pred, dtype=np.float32)
        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1

        h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred