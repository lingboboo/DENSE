
from typing import List
from typing import List
import logging
from pathlib import Path
from typing import Tuple, Union
from scipy.spatial import cKDTree

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from skimage.color import rgba2rgb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index

from base_ml.base_early_stopping import EarlyStopping
from base_ml.base_trainer import BaseTrainer
from models.segmentation.nuclei_segmentation.Dense import DataclassHVStorage
from nuclei_segmentation.utils.metrics import get_fast_pq, remap_label
from nuclei_segmentation.utils.tools import cropping_center
from models.segmentation.nuclei_segmentation.Dense import Dense
from utils.tools import AverageMeter


online_open_overlay = True
online_epoch = 20
offline_open = True
iou_open = True
class DenseTrainer(BaseTrainer):
    def __init__(
        self,
        model: Dense,
        loss_fn_dict: dict,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        dataset_config: dict,
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        log_images: bool = False,
        magnification: int = 40,
        mixed_precision: bool = False,
    ):
        super().__init__(
            model=model,
            loss_fn=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            accum_iter=1,
            log_images=log_images,
            mixed_precision=mixed_precision,
        )
        self.loss_fn_dict = loss_fn_dict
        self.dataset_config = dataset_config
        self.magnification = magnification

        self.loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss", ":.4f")}
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"] = AverageMeter(
                    f"{branch}_{loss_name}", ":.4f"
                )

    def train_epoch(
        self, epoch: int, train_dataloader: DataLoader, unfreeze_epoch: int = 50
    ) -> Tuple[dict, dict]:
        self.model.train()
        if epoch >= unfreeze_epoch:
            self.model.unfreeze_encoder()

        binary_dice_scores = []
        binary_jaccard_scores = []
        train_example_img = None
        branch_losses_sum = {}
        branch_losses_count = {}
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()

        if self.log_images:
            select_example_image = int(torch.randint(0, len(train_dataloader), (1,)))
        else:
            select_example_image = None
        train_loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, batch in train_loop:
            return_example_images = batch_idx == select_example_image
            batch_metrics, example_img, branch_losses = self.train_step(
                batch,
                batch_idx,
                len(train_dataloader),
                return_example_images=return_example_images,
                epoch=epoch,
            )
            if example_img is not None:
                train_example_img = example_img
            binary_dice_scores = binary_dice_scores + batch_metrics["binary_dice_scores"]
            binary_jaccard_scores = (
                binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
            )

            for k, v in branch_losses.items():
                v_item = v.item() if hasattr(v, "item") else float(v)
                if k not in branch_losses_sum:
                    branch_losses_sum[k] = 0.0
                    branch_losses_count[k] = 0
                branch_losses_sum[k] += v_item
                branch_losses_count[k] += 1

            train_loop.set_postfix(
                {
                    "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                    "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                }
            )

        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)

        branch_losses_avg = {}
        for k in branch_losses_sum:
            branch_losses_avg[k] = branch_losses_sum[k] / (branch_losses_count[k] + 1e-8)
        loss_strs = [f"{k}: {v:.4f}" for k, v in branch_losses_avg.items()]
        loss_str_joined = " - ".join(loss_strs)

        scalar_metrics = {
            "Loss/Train": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-nuclei-Dice-Mean/Train": np.nanmean(binary_dice_scores),
            "Binary-nuclei-Jacard-Mean/Train": np.nanmean(binary_jaccard_scores),
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[f"{branch}_{loss_name}/Train"] = self.loss_avg_tracker[
                    f"{branch}_{loss_name}"
                ].avg

        self.logger.info(
            f"{'Training epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Binary-nuclei-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-nuclei-Jacard: {np.nanmean(binary_jaccard_scores):.4f} - "
            f"{loss_str_joined}"
        )

        image_metrics = {"Example-Predictions/Train": train_example_img}

        return scalar_metrics, image_metrics

    def train_step(
        self,
        batch: object,
        batch_idx: int,
        num_batches: int,
        return_example_images: bool,
        epoch:int,
    ) -> Tuple[dict, Union[plt.Figure, None]]:

        imgs = batch[0].to(self.device)
        masks = batch[1]
        filenames = batch[2]
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions_ = self.model.forward(imgs)
                predictions = self.unpack_predictions(epoch=epoch,masks=masks, predictions=predictions_, filenames=filenames)
                gt, uncertain_mask = self.unpack_masks(epoch=epoch, predictions=predictions, masks=masks, filenames=filenames,is_train=True)
                total_loss, branch_losses= self.calculate_loss(predictions, gt, uncertain_mask)
                self.scaler.scale(total_loss).backward()
                if (
                    ((batch_idx + 1) % self.accum_iter == 0)
                    or ((batch_idx + 1) == num_batches)
                    or (self.accum_iter == 1)
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()
        else:
            predictions_ = self.model.forward(imgs)
            predictions = self.unpack_predictions(epoch=epoch,masks=masks,predictions=predictions_,filenames=filenames)
            gt, uncertain_mask = self.unpack_masks(epoch=epoch, predictions=predictions, masks=masks, filenames=filenames,is_train=True)
            total_loss, branch_losses= self.calculate_loss(predictions, gt, uncertain_mask)
            total_loss.backward()
            if (
                ((batch_idx + 1) % self.accum_iter == 0)
                or ((batch_idx + 1) == num_batches)
                or (self.accum_iter == 1)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

        batch_metrics = self.calculate_step_metric_train(predictions, gt)
        if return_example_images:
            return_example_images = self.generate_example_image(
                imgs, predictions, gt, num_images=4
            )
        else:
            return_example_images = None

        return batch_metrics, return_example_images, branch_losses

    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float]:
        self.model.eval()

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        val_example_img = None

        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()

        if self.log_images:
            select_example_image = int(torch.randint(0, len(val_dataloader), (1,)))
        else:
            select_example_image = None

        val_loop = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                return_example_images = batch_idx == select_example_image
                batch_metrics, example_img = self.validation_step(
                    batch, batch_idx, return_example_images,epoch
                )
                if example_img is not None:
                    val_example_img = example_img
                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                val_loop.set_postfix(
                    {
                        "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                        "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                        "acc": np.round(np.nanmean(binary_jaccard_scores), 3),
                        "pq": np.round(np.nanmean(pq_scores), 3),
                    }
                )

        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)

        scalar_metrics = {
            "Loss/Validation": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-nuclei-Dice-Mean/Validation": np.nanmean(binary_dice_scores),
            "Binary-nuclei-Jacard-Mean/Validation": np.nanmean(binary_jaccard_scores),
            "bPQ/Validation": np.nanmean(pq_scores),
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[
                    f"{branch}_{loss_name}/Validation"
                ] = self.loss_avg_tracker[f"{branch}_{loss_name}"].avg

        self.logger.info(
            f"{'Validation epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Binary-nuclei-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-nuclei-Jacard: {np.nanmean(binary_jaccard_scores):.4f} - "
            f"bPQ-Score: {np.nanmean(pq_scores):.4f} - "
        )

        image_metrics = {"Example-Predictions/Validation": val_example_img}

        return scalar_metrics, image_metrics, np.nanmean(pq_scores)

    def validation_step(
        self,
        batch: object,
        batch_idx: int,
        return_example_images: bool,
        epoch:int,
    ):
        imgs = batch[0].to(self.device)
        masks = batch[1]
        filenames = batch[2]
        self.model.zero_grad()
        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions_ = self.model.forward(imgs)
                predictions = self.unpack_predictions(epoch=epoch,masks=masks,predictions=predictions_, filenames=filenames)
                gt, uncertain_mask = self.unpack_masks(epoch=epoch, predictions=predictions, masks=masks, filenames=filenames,is_train=False)
                _,_ = self.calculate_loss(predictions, gt, uncertain_mask)
        else:
            predictions_ = self.model.forward(imgs)
            predictions = self.unpack_predictions(epoch=epoch,masks=masks,predictions=predictions_,filenames=filenames)
            gt, uncertain_mask = self.unpack_masks(epoch=epoch, predictions=predictions, masks=masks, filenames=filenames,is_train=False)
            _ ,_= self.calculate_loss(predictions, gt, uncertain_mask)
        
        batch_metrics = self.calculate_step_metric_validation(predictions, gt)
                
        if return_example_images:
            try:
                return_example_images = self.generate_example_image(
                    imgs,
                    predictions,
                    gt,
                    num_images=4,
                )
            except AssertionError:
                self.logger.error(
                    "AssertionError for Example Image. Please check. Continue without image."
                )
                return_example_images = None
        else:
            return_example_images = None

        return batch_metrics, return_example_images

    def unpack_predictions(self, epoch:int, masks: dict, predictions: dict, filenames: List[str], save_train_path: str) -> DataclassHVStorage:
        uncertain_mask = masks["uncertain_mask"]
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
        predictions["instance_map"], _ = self.model.calculate_instance_map(predictions, self.magnification)

        if "regression_map" not in predictions.keys():
            predictions["regression_map"] = None
        if "den_map" not in predictions.keys():
            predictions["den_map"] = None
            den_loss=False
        else:
            den_loss=True
            predictions["den_map"]=torch.relu(predictions["den_map"])

        predictions = DataclassHVStorage(
            nuclei_binary_map=predictions["nuclei_binary_map"],
            hv_map=predictions["hv_map"],
            instance_map=predictions["instance_map"],
            batch_size=predictions["nuclei_binary_map"].shape[0],
            regression_map=predictions["regression_map"],
            den_map = predictions["den_map"],
            density_loss = den_loss,  
        )
        
        return predictions

    def unpack_masks(self, epoch:int, predictions:DataclassHVStorage , masks: dict,filenames: List[str],is_train:bool) -> DataclassHVStorage:
        predictions = predictions.get_dict()
        if is_train:
            gt_nuclei_binary_map = masks["pse_nuclei_binary_map"].to(self.device)
            gt_nuclei_binary_map_onehot = (
                F.one_hot(masks["pse_nuclei_binary_map"], num_classes=2)
            ).type(
                torch.float32
            ) 
            gt = {
                "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                    self.device
                ),
                "hv_map": masks["pse_hv_map"].to(self.device),
                "instance_map": masks["pse_instance_map"].to(
                    self.device
                ),
            }
        else:
            gt_nuclei_binary_map = masks["gt_nuclei_binary_map"].to(self.device)

            gt_nuclei_binary_map_onehot = (
                F.one_hot(masks["gt_nuclei_binary_map"], num_classes=2)
            ).type(
                torch.float32
            ) 
            gt = {
                "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                    self.device
                ),
                "hv_map": masks["gt_hv_map"].to(self.device),
                "instance_map": masks["gt_instance_map"].to(
                    self.device
                ),
            }            
        if "mask_knn" in masks:
            gt_mask_knn_onehot = (
                F.one_hot(masks["mask_knn"], num_classes=2)
            ).type(torch.float32)
            gt["mask_knn"] = gt_mask_knn_onehot.permute(0, 3, 1, 2).to(
                self.device
            ) 
        if "regression_map" in masks:
            gt["regression_map"] = masks["regression_map"].to(self.device)
        if "den_map" in masks: 
            gt["den_map"] = masks["den_map"].unsqueeze(1).to(self.device) 
            gt["point_map"] = masks["point_map"].to(self.device)
            den_loss=True
            if online_open_overlay and epoch > online_epoch and "den_map" in predictions :
                den_map_squeezed = predictions["den_map"][:, 0, :, :].clone().to(self.device)   
                masks["uncertain_mask"] = masks["uncertain_mask"].to(self.device)  
                uncertain_mask = masks["uncertain_mask"]
                nuclei_condition = den_map_squeezed > 0  
                uncertain_condition = predictions["den_map"] > 0.1  
                gt_nuclei_binary_map[nuclei_condition]=1
                uncertain_mask[uncertain_condition] = 1
                den_map_squeezed = predictions["den_map"][:, 0, :, :].clone().to(self.device)   
                pred_den_binary = (den_map_squeezed > 0).float()  
                if is_train:
                    original_semantic = masks["pse_nuclei_binary_map"].float().to(self.device)
                else:
                    original_semantic = masks["gt_nuclei_binary_map"].float().to(self.device)
                new_semantic = torch.clamp(original_semantic + pred_den_binary, max=1.0)
                new_foreground = (new_semantic == 1) & (original_semantic == 0)  
                if offline_open:
                    gt_nuclei_binary_map[masks["mask_knn"]==0]=0
                    mask_knn = masks["mask_knn"].unsqueeze(1)  
                    masks["uncertain_mask"][mask_knn==0]=1
                gt_nuclei_binary_map_onehot = (F.one_hot(gt_nuclei_binary_map, num_classes=2)).type(torch.float32).permute(0, 3, 1, 2).to(self.device) 
                gt["nuclei_binary_map"] = gt_nuclei_binary_map_onehot
                hv_map = gt["hv_map"]  
                point_map = masks["point_map"].to(self.device)  
                updated_hv_map = self.update_hv_map(hv_map, point_map, new_foreground)  
                gt["hv_map"] = updated_hv_map  
                gt["instance_map"], _ = self.model.calculate_instance_map(gt, self.magnification) 
            else:
                uncertain_mask = masks["uncertain_mask"]

        gt = DataclassHVStorage(
            **gt,
            batch_size=gt["nuclei_binary_map"].shape[0],
            density_loss=den_loss,
        )
        return gt, uncertain_mask

    def update_hv_map(self, hv_map: torch.Tensor, point_map: torch.Tensor, new_foreground: torch.Tensor) -> torch.Tensor:
        B, C, H, W = hv_map.shape
        assert C == 2, "hv_map 应该有2个通道，分别表示水平和垂直距离"
        updated_hv_map = hv_map.clone()

        for b in range(B):
            new_fg_indices = torch.nonzero(new_foreground[b], as_tuple=False).cpu().numpy()
            if new_fg_indices.size == 0:
                continue

            point_coords = torch.nonzero(point_map[b] == 255, as_tuple=False).cpu().numpy()
            if point_coords.size == 0:
                continue

            tree = cKDTree(point_coords)
            distances, indices = tree.query(new_fg_indices, k=1)
            nearest_points = point_coords[indices]

            delta_y = new_fg_indices[:, 0] - nearest_points[:, 0]
            delta_x = new_fg_indices[:, 1] - nearest_points[:, 1]

            for i, (y, x) in enumerate(new_fg_indices):
                dist = distances[i]
                if dist > 32.0:
                    continue
                else:
                    norm_x = delta_x[i] / (W / 2.0)
                    norm_y = delta_y[i] / (H / 2.0)
                    norm_x = np.clip(norm_x, -1.0, 1.0)
                    norm_y = np.clip(norm_y, -1.0, 1.0)

                    updated_hv_map[b, 0, y, x] = norm_x
                    updated_hv_map[b, 1, y, x] = norm_y

        return updated_hv_map
    
    def calculate_loss(
        self, predictions: DataclassHVStorage, gt: DataclassHVStorage, uncertain_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        predictions = predictions.get_dict()
        gt = gt.get_dict()
        uncertain_mask=uncertain_mask.to(self.device)

        total_loss = 0
        branch_losses = {}
        for branch, pred in predictions.items():
            if branch in ["instance_map","point_map"]:
                continue
            if branch not in self.loss_fn_dict:
                continue
            branch_loss_fns = self.loss_fn_dict[branch]
            for loss_name, loss_setting in branch_loss_fns.items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name == "msge":
                    loss_value = loss_fn(
                        input=pred,
                        target=gt[branch],
                        focus=gt["nuclei_binary_map"],
                        device=self.device,
                    )
                else:
                    loss_value = loss_fn(input=pred, target=gt[branch],filtered_mask=uncertain_mask)
                weighted_loss = weight * loss_value
                total_loss += weighted_loss
                self.loss_avg_tracker[f"{branch}_{loss_name}"].update(
                    loss_value.detach().cpu().numpy()
                )
                branch_losses[f"{branch}_{loss_name}"] = loss_value.detach().cpu().item()

        if iou_open and predictions["den_map"] is not None:
            den_binary_map = (predictions["den_map"] > 0).float().squeeze(1)
            gt_foreground = gt["nuclei_binary_map"][:, 1, :, :].float()
            intersection = (den_binary_map * gt_foreground).sum(dim=(1, 2))
            union = (den_binary_map + gt_foreground).clamp(max=1.0).sum(dim=(1, 2))
            iou_loss = 1.0 - (intersection / (union + 1e-8)).mean()
            lambda_iou = 1.0
            weighted_iou_loss = lambda_iou * iou_loss
            total_loss += weighted_iou_loss
            branch_losses["iou_loss"] = iou_loss.detach().cpu().item()     
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())

        return total_loss, branch_losses

    def calculate_step_metric_train(
        self, predictions: DataclassHVStorage, gt: DataclassHVStorage
    ) -> dict:
        predictions = predictions.get_dict()
        gt = gt.get_dict()

        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )

        binary_dice_scores = []
        binary_jaccard_scores = []

        for i in range(gt["nuclei_binary_map"].shape[0]):
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            nuclei_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(nuclei_dice))

            nuclei_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(nuclei_jaccard))

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
        }

        return batch_metrics

    def calculate_step_metric_validation(self, predictions: dict, gt: dict) -> dict:

        predictions = predictions.get_dict()
        gt = gt.get_dict()

        predictions["instance_map"] = predictions["instance_map"].detach().cpu()

        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []

        for i in range(gt["nuclei_binary_map"].shape[0]):
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            nuclei_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(nuclei_dice))

            nuclei_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(nuclei_jaccard))
            remapped_instance_pred = remap_label(predictions["instance_map"][i])
            remapped_gt = remap_label(instance_maps_gt[i])
            [dq, sq, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)
            
        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
        }

        return batch_metrics

    @staticmethod
    def generate_example_image(
        imgs: Union[torch.Tensor, np.ndarray],
        predictions: DataclassHVStorage,
        gt: DataclassHVStorage,
        num_images: int = 2,
    ) -> plt.Figure:

        predictions = predictions.get_dict()
        gt = gt.get_dict()

        assert num_images <= imgs.shape[0]
        num_images = 4

        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)

        h = gt["hv_map"].shape[1]
        w = gt["hv_map"].shape[2]

        sample_indices = torch.randint(0, imgs.shape[0], (num_images,))
        sample_images = (
            imgs[sample_indices].permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )
        sample_images = cropping_center(sample_images, (h, w), True)

        pred_sample_binary_map = (
            predictions["nuclei_binary_map"][sample_indices, :, :, 1]
            .detach()
            .cpu()
            .numpy()
        )
        pred_sample_hv_map = (
            predictions["hv_map"][sample_indices].detach().cpu().numpy()
        )
        pred_sample_instance_maps = (
            predictions["instance_map"][sample_indices].detach().cpu().numpy()
        )

        gt_sample_binary_map = (
            gt["nuclei_binary_map"][sample_indices].detach().cpu().numpy()
        )
        gt_sample_hv_map = gt["hv_map"][sample_indices].detach().cpu().numpy()
        gt_sample_instance_map = (
            gt["instance_map"][sample_indices].detach().cpu().numpy()
        )

        hv_cmap = plt.get_cmap("jet")
        binary_cmap = plt.get_cmap("jet")
        instance_map = plt.get_cmap("viridis")

        fig, axs = plt.subplots(num_images, figsize=(6, 2 * num_images), dpi=150)

        for i in range(num_images):
            placeholder = np.zeros((2 * h, 6 * w, 3))
            placeholder[:h, :w, :3] = sample_images[i]
            placeholder[h : 2 * h, :w, :3] = sample_images[i]
            placeholder[:h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(gt_sample_binary_map[i] * 255)
            )
            placeholder[h : 2 * h, w : 2 * w, :3] = rgba2rgb(
                binary_cmap(pred_sample_binary_map[i])
            )
            placeholder[:h, 2 * w : 3 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, :, :, 0] + 1) / 2)
            )
            placeholder[h : 2 * h, 2 * w : 3 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, :, :, 0] + 1) / 2)
            )
            placeholder[:h, 3 * w : 4 * w, :3] = rgba2rgb(
                hv_cmap((gt_sample_hv_map[i, :, :, 1] + 1) / 2)
            )
            placeholder[h : 2 * h, 3 * w : 4 * w, :3] = rgba2rgb(
                hv_cmap((pred_sample_hv_map[i, :, :, 1] + 1) / 2)
            )
            placeholder[:h, 4 * w : 5 * w, :3] = rgba2rgb(
                instance_map(
                    (gt_sample_instance_map[i] - np.min(gt_sample_instance_map[i]))
                    / (
                        np.max(gt_sample_instance_map[i])
                        - np.min(gt_sample_instance_map[i] + 1e-10)
                    )
                )
            )
            placeholder[h : 2 * h, 4 * w : 5 * w, :3] = rgba2rgb(
                instance_map(
                    (
                        pred_sample_instance_maps[i]
                        - np.min(pred_sample_instance_maps[i])
                    )
                    / (
                        np.max(pred_sample_instance_maps[i])
                        - np.min(pred_sample_instance_maps[i] + 1e-10)
                    )
                )
            )

            axs[i].imshow(placeholder)
            axs[i].set_xticks([], [])

            if i == 0:
                axs[i].set_xticks(np.arange(w / 2, 6 * w, w))
                axs[i].set_xticklabels(
                    [
                        "Image",
                        "Binary-nucleis",
                        "HV-Map-0",
                        "HV-Map-1",
                        "nuclei Instances",
                        "Nuclei-Instances",
                    ],
                    fontsize=6,
                )
                axs[i].xaxis.tick_top()

            axs[i].set_yticks(np.arange(h / 2, 2 * h, h))
            axs[i].set_yticklabels(["GT", "Pred."], fontsize=6)
            axs[i].tick_params(axis="both", which="both", length=0)
            grid_x = np.arange(w, 6 * w, w)
            grid_y = np.arange(h, 2 * h, h)

            for x_seg in grid_x:
                axs[i].axvline(x_seg, color="black")
            for y_seg in grid_y:
                axs[i].axhline(y_seg, color="black")

        fig.suptitle(f"Patch Predictions for {num_images} Examples")
        fig.tight_layout()

        return fig