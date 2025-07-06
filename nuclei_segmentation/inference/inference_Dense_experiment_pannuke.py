import argparse
import inspect
import os
import sys
from scipy.optimize import linear_sum_assignment
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from base_ml.base_experiment import BaseExperiment

BaseExperiment.seed_run(1232)

import json
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms

from nuclei_segmentation.datasets.dataset_coordinator import select_dataset
from models.segmentation.nuclei_segmentation.Dense import DataclassHVStorage
from nuclei_segmentation.utils.metrics import (
    get_fast_pq,
    remap_label,
)
from nuclei_segmentation.utils.tools import cropping_center
from models.segmentation.nuclei_segmentation.Dense import Dense

from utils.logger import Logger

save_inference_images = True

class InferenceDense:
    def __init__(self, run_dir: Union[Path, str], gpu: int, magnification: int = 40, checkpoint_name: str = "latest_checkpoint.pth") -> None:
        self.run_dir = Path(run_dir)
        self.device = f"cuda:{gpu}"
        self.run_conf: dict = None
        self.logger: Logger = None
        self.magnification = magnification
        self.checkpoint_name = checkpoint_name

        self.__load_run_conf()
        self.__load_dataset_setup(dataset_path=self.run_conf["data"]["dataset_path"])
        self.__instantiate_logger()
        self.__check_eval_model()
        self.__setup_amp()

        self.logger.info(f"Loaded run: {run_dir}")

    def __load_run_conf(self) -> None:
        with open((self.run_dir / "config.yaml").resolve(), "r") as run_config_file:
            yaml_config = yaml.safe_load(run_config_file)
            self.run_conf = dict(yaml_config)

    def __load_dataset_setup(self, dataset_path: Union[Path, str]) -> None:
        dataset_config_path = Path(dataset_path) / "dataset_config.yaml"
        with open(dataset_config_path, "r") as dataset_config_file:
            yaml_config = yaml.safe_load(dataset_config_file)
            self.dataset_config = dict(yaml_config)

    def __instantiate_logger(self) -> None:
        logger = Logger(
            level=self.run_conf["logging"]["level"].upper(),
            log_dir=Path(self.run_dir).resolve(),
            comment="inference",
            use_timestamp=False,
            formatter="%(message)s",
        )
        self.logger = logger.create_logger()

    def __check_eval_model(self) -> None:
        assert (self.run_dir / "checkpoints" / self.checkpoint_name).is_file(), (
            f"Checkpoint file {self.checkpoint_name} does not exist in {self.run_dir / 'checkpoints'}"
        )

    def __setup_amp(self) -> None:
        self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)

    def get_model(self, model_type: str) -> Union[Dense]:
        implemented_models = ["Dense"]
        if model_type not in implemented_models:
            raise NotImplementedError(f"Unknown model type. Please select one of {implemented_models}")

        model_class = Dense
        model = model_class(
            embed_dim=self.run_conf["model"]["embed_dim"],
            input_channels=self.run_conf["model"].get("input_channels", 3),
            depth=self.run_conf["model"]["depth"],
            num_heads=self.run_conf["model"]["num_heads"],
            extract_layers=self.run_conf["model"]["extract_layers"],
            regression_loss=self.run_conf["model"].get("regression_loss", False),
            den_loss=self.run_conf["model"].get("den_loss", False),
        )

        return model

    def setup_patch_inference(self, test_folds: List[int] = None) -> tuple[Dense, DataLoader, dict]:
        checkpoint = torch.load(self.run_dir / "checkpoints" / self.checkpoint_name, map_location="cpu")
        model = self.get_model(model_type=checkpoint["arch"])
        self.logger.info(f"Loading best model from {str(self.run_dir / 'checkpoints' / self.checkpoint_name)}")
        self.logger.info(model.load_state_dict(checkpoint["model_state_dict"]))

        if test_folds is None:
            if "test_folds" in self.run_conf["data"]:
                if self.run_conf["data"]["test_folds"] is None:
                    self.logger.info("There was no test set provided. We now use the validation dataset for testing")
                    self.run_conf["data"]["test_folds"] = self.run_conf["data"]["val_folds"]
            else:
                self.logger.info("There was no test set provided. We now use the validation dataset for testing")
                self.run_conf["data"]["test_folds"] = self.run_conf["data"]["val_folds"]
        else:
            self.run_conf["data"]["test_folds"] = self.run_conf["data"]["val_folds"]
        self.logger.info(f"Performing Inference on test set: {self.run_conf['data']['test_folds']}")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        transforms = A.Compose([A.Normalize(mean=mean, std=std)])

        inference_dataset = select_dataset(
            dataset_name=self.run_conf["data"]["dataset"],
            split="test",
            dataset_config=self.run_conf["data"],
            transforms=transforms,
        )

        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=16,
            num_workers=12,
            pin_memory=False,
            shuffle=False,
        )

        return model, inference_dataloader, self.dataset_config

    def run_patch_inference(
        self,
        model: Dense,
        inference_dataloader: DataLoader,
        dataset_config: dict,
        generate_plots: bool = False,
    ) -> None:
        model.to(device=self.device)
        model.eval()

        image_names = []
        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        f1_scores = []
        accuracy_scores = []
        aji_scores = []

        inference_loop = tqdm.tqdm(enumerate(inference_dataloader), total=len(inference_dataloader))

        with torch.no_grad():
            for batch_idx, batch in inference_loop:
                batch_metrics = self.inference_step(model, batch, generate_plots=generate_plots)
                image_names = image_names + batch_metrics["image_names"]
                binary_dice_scores = binary_dice_scores + batch_metrics["binary_dice_scores"]
                binary_jaccard_scores = binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                dq_scores = dq_scores + batch_metrics["dq_scores"]
                sq_scores = sq_scores + batch_metrics["sq_scores"]
                f1_scores = f1_scores + batch_metrics["f1_scores"]
                accuracy_scores = accuracy_scores + batch_metrics["accuracy_scores"]
                aji_scores = aji_scores + batch_metrics["aji_scores"]

        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        dq_scores = np.array(dq_scores)
        sq_scores = np.array(sq_scores)
        f1_scores = np.array(f1_scores)
        accuracy_scores = np.array(accuracy_scores)
        aji_scores = np.array(aji_scores)

        dataset_metrics = {
            "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
            "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
            "bPQ": float(np.nanmean(pq_scores)),
            "bDQ": float(np.nanmean(dq_scores)),
            "bSQ": float(np.nanmean(sq_scores)),
            "F1": float(np.nanmean(f1_scores)),
            "ACC": float(np.nanmean(accuracy_scores)),
            "AJI": float(np.nanmean(aji_scores)),
        }

        self.logger.info(f"{20*'*'} Binary Dataset metrics {20*'*'}")
        [self.logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]

        image_metrics = {}
        for idx, image_name in enumerate(image_names):
            image_metrics[image_name] = {
                "Dice": float(binary_dice_scores[idx]),
                "Jaccard": float(binary_jaccard_scores[idx]),
                "bPQ": float(pq_scores[idx]),
            }
        all_metrics = {
            "dataset": dataset_metrics,
            "image_metrics": image_metrics,
        }

        with open(str(self.run_dir / "inference_results.json"), "w") as outfile:
            json.dump(all_metrics, outfile, indent=2)

    def inference_step(
        self,
        model: Dense,
        batch: tuple,
        generate_plots: bool = False,
    ) -> None:
        imgs = batch[0].to(self.device)
        masks = batch[1]
        image_names = list(batch[2])

        model.zero_grad()
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model.forward(imgs)
        else:
            predictions = model.forward(imgs)
        predictions = self.unpack_predictions(predictions=predictions, model=model)
        gt = self.unpack_masks(masks=masks, model=model)

        batch_metrics, scores = self.calculate_step_metric(predictions, gt, image_names)

        if save_inference_images:
            try:
                self.save_predictions_and_gt(
                    predictions=predictions,
                    gt=gt,
                    filenames=image_names,
                    save_dir=Path(self.run_dir / "inference_predictions")
                )
            except Exception as e:
                self.logger.error(
                    f"Error while saving train images: {e}. Continue without saving images."
                )

        if generate_plots:
            self.plot_results(
                imgs=imgs,
                predictions=predictions,
                ground_truth=gt,
                img_names=image_names,
                outdir=Path(self.run_dir / "inference_predictions"),
                scores=scores,
            )

        return batch_metrics
def unpack_predictions(self, predictions: dict, model: Dense) -> DataclassHVStorage:
    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
    predictions["instance_map"], _ = model.calculate_instance_map(predictions, magnification=self.magnification)

    if "regression_map" not in predictions.keys():
        predictions["regression_map"] = None
    if "den_map" not in predictions.keys():
        predictions["den_map"] = None

    predictions = DataclassHVStorage(
        nuclei_binary_map=predictions["nuclei_binary_map"],
        hv_map=predictions["hv_map"],
        instance_map=predictions["instance_map"],
        batch_size=predictions["nuclei_binary_map"].shape[0],
        regression_map=predictions["regression_map"],
        den_map=predictions["den_map"],
    )

    return predictions

def unpack_masks(self, masks: dict, model: Dense) -> DataclassHVStorage:
    gt_nuclei_binary_map_onehot = (
        F.one_hot(masks["gt_nuclei_binary_map"], num_classes=2)
    ).type(torch.float32)

    gt = {
        "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(self.device),
        "hv_map": masks["gt_hv_map"].to(self.device),
        "instance_map": masks["gt_instance_map"].to(self.device),
    }

    if "regression_map" in masks:
        gt["regression_map"] = masks["regression_map"].to(self.device)
    if "den_map" in masks:
        gt["den_map"] = masks["den_map"].to(self.device)

    gt = DataclassHVStorage(**gt, batch_size=gt["nuclei_binary_map"].shape[0])

    return gt

def calculate_step_metric(self, predictions: DataclassHVStorage, gt: DataclassHVStorage, image_names: list[str]) -> Tuple[dict, list]:
    predictions = predictions.get_dict()
    gt = gt.get_dict()

    predictions["instance_map"] = predictions["instance_map"].detach().cpu()
    instance_maps_gt = gt["instance_map"].detach().cpu()
    gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(torch.uint8)

    binary_dice_scores = []
    binary_jaccard_scores = []
    pq_scores = []
    dq_scores = []
    sq_scores = []
    accuracy_scores = []
    f1_scores = []
    scores = []
    aji_scores = []

    for i in range(gt["nuclei_binary_map"].shape[0]):
        pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
        target_binary_map = gt["nuclei_binary_map"][i]
        cell_dice = dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0).detach().cpu()
        binary_dice_scores.append(float(cell_dice))

        cell_jaccard = binary_jaccard_index(preds=pred_binary_map, target=target_binary_map).detach().cpu()
        binary_jaccard_scores.append(float(cell_jaccard))

        if len(np.unique(instance_maps_gt[i])) == 1:
            dq, sq, pq = np.nan, np.nan, np.nan
        else:
            remapped_instance_pred = remap_label(predictions["instance_map"][i])
            remapped_gt = remap_label(instance_maps_gt[i])
            [dq, sq, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
        pq_scores.append(pq)
        dq_scores.append(dq)
        sq_scores.append(sq)

        accuracy = self.calculate_accuracy(pred_binary_map, target_binary_map)
        accuracy_scores.append(float(accuracy))

        f1_score = self.calculate_f1_score(pred_binary_map, target_binary_map)
        f1_scores.append(float(f1_score))

        scores.append([cell_dice.detach().cpu().numpy(), cell_jaccard.detach().cpu().numpy(), pq, accuracy, f1_score])

        remapped_instance_pred = remap_label(predictions["instance_map"][i])
        remapped_gt = remap_label(instance_maps_gt[i])
        aji_score = self.get_fast_aji_plus(true=remapped_gt, pred=remapped_instance_pred)
        aji_scores.append(aji_score)

    batch_metrics = {
        "image_names": image_names,
        "binary_dice_scores": binary_dice_scores,
        "binary_jaccard_scores": binary_jaccard_scores,
        "pq_scores": pq_scores,
        "dq_scores": dq_scores,
        "sq_scores": sq_scores,
        "accuracy_scores": accuracy_scores,
        "f1_scores": f1_scores,
        "aji_scores": aji_scores,
    }

    return batch_metrics, scores

def get_fast_aji_plus(self, true, pred):
    true = remap_label(true)
    pred = remap_label(pred)

    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    pairwise_inter = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)

    paired_iou = pairwise_iou[paired_true, paired_pred]
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]

    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()

    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score

def calculate_accuracy(self, pred_map, gt_map):
    TP = torch.sum((pred_map == 1) & (gt_map == 1)).float()
    TN = torch.sum((pred_map == 0) & (gt_map == 0)).float()
    FP = torch.sum((pred_map == 1) & (gt_map == 0)).float()
    FN = torch.sum((pred_map == 0) & (gt_map == 1)).float()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def calculate_f1_score(self, pred_map, gt_map):
    TP = torch.sum((pred_map == 1) & (gt_map == 1)).float()
    FP = torch.sum((pred_map == 1) & (gt_map == 0)).float()
    FN = torch.sum((pred_map == 0) & (gt_map == 1)).float()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def plot_results(self, imgs: Union[torch.Tensor, np.ndarray], predictions: dict, ground_truth: dict, img_names: List, outdir: Union[Path, str], scores: List[List[float]] = None) -> None:
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    h = ground_truth["hv_map"].shape[1]
    w = ground_truth["hv_map"].shape[2]

    sample_images = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    sample_images = cropping_center(sample_images, (h, w), True)

    pred_sample_hv_map = predictions["hv_map"].detach().cpu().numpy()
    gt_sample_hv_map = ground_truth["hv_map"].detach().cpu().numpy()
    hv_cmap = plt.get_cmap("jet")

    transform_settings = self.run_conf["transformations"]
    if "normalize" in transform_settings:
        mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
        std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    inv_normalize = transforms.Normalize(
        mean=[-0.5 / mean[0], -0.5 / mean[1], -0.5 / mean[2]],
        std=[1 / std[0], 1 / std[1], 1 / std[2]],
    )
    inv_samples = inv_normalize(torch.tensor(sample_images).permute(0, 3, 1, 2))
    sample_images = inv_samples.permute(0, 2, 3, 1).detach().cpu().numpy()

    for i in range(len(img_names)):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(sample_images[i], cmap='gray')
        hv_map_overlay = np.sum(pred_sample_hv_map[i], axis=-1)
        ax.imshow(hv_map_overlay, cmap=hv_cmap, alpha=0.5)
        out_filepath = outdir / f"{img_names[i]}.png"
        plt.savefig(out_filepath)
        plt.close()


class InferenceDenseParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform Dense inference for given run-directory with model checkpoints and logs",
        )
        parser.add_argument("--run_dir", type=str, help="Logging directory of a training run.", required=True)
        parser.add_argument("--checkpoint_name", type=str, help="Name of the checkpoint. Defaults to 'model_best.pth'", default="model_best.pth")
        parser.add_argument("--gpu", type=int, help="Cuda-GPU ID for inference", default=5)
        parser.add_argument("--magnification", type=int, help="Dataset Magnification. Defaults to 40", choices=[20, 40], default=40)
        parser.add_argument("--plots", action="store_true", help="Generate inference plots in run_dir")

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)

if __name__ == "__main__":
    configuration_parser = InferenceDenseParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)
    inf = InferenceDense(
        run_dir=configuration["run_dir"],
        checkpoint_name=configuration["checkpoint_name"],
        gpu=configuration["gpu"],
        magnification=configuration["magnification"],
    )
    model, dataloader, conf = inf.setup_patch_inference()

    inf.run_patch_inference(model, dataloader, conf, generate_plots=configuration["plots"])
