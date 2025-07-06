import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch import nn
from torch.nn.modules.loss import _Loss
from base_ml.base_utils import filter2D, gaussian_kernel2d
class DensityMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        filtered_mask: torch.Tensor = None, 
    ) -> torch.Tensor:
        assert input.shape == target.shape, "Input and target must have the same shape."
        assert input.shape[1] == 1, "Density maps should have one channel."

        loss = F.mse_loss(input, target, reduction='mean')
        return loss
    
class XentropyLoss(_Loss):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        filtered_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        epsilon = 1e-8
        pred = input / (torch.sum(input, dim=-1, keepdim=True) + epsilon)
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

        ce = -torch.sum(target * torch.log(pred), dim=-1, keepdim=True)

        if filtered_mask is not None:
            mask_4d = filtered_mask.permute(0, 2, 3, 1)
            ce = ce * mask_4d
            if self.reduction == "mean":
                valid_count = mask_4d.sum()
                loss = ce.sum() / (valid_count + 1e-8)
            else:
                loss = ce.sum()
        else:
            loss = ce.mean() if self.reduction == "mean" else ce.sum()

        return loss

class DiceLoss(_Loss):
    def __init__(self, smooth: float = 1e-3) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")
        self.smooth = smooth

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        filtered_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        if filtered_mask is not None:
            mask_4d = filtered_mask.permute(0, 2, 3, 1)
            mask_4d = mask_4d.expand(-1, -1, -1, input.shape[-1])
            input = input * mask_4d
            target = target * mask_4d

        inse = torch.sum(input * target, (0, 1, 2))
        l = torch.sum(input, (0, 1, 2))
        r = torch.sum(target, (0, 1, 2))

        loss = 1.0 - (2.0 * inse + self.smooth) / (l + r + self.smooth)
        loss = torch.sum(loss)

        return loss


class MSELossMaps(_Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        filtered_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        diff = (input - target) ** 2
        diff = diff.mean(dim=(2,3))
        diff = diff.mean(dim=1)
        loss = diff.mean()
        return loss



class MSGELossMaps(_Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def get_sobel_kernel(
        self, size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range, indexing="ij")
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(self, hv: torch.Tensor, device: str) -> torch.Tensor:
        kernel_h, kernel_v = self.get_sobel_kernel(5, device=device)
        kernel_h = kernel_h.view(1, 1, 5, 5)
        kernel_v = kernel_v.view(1, 1, 5, 5)

        h_ch = hv[..., 0].unsqueeze(1)
        v_ch = hv[..., 1].unsqueeze(1)

        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()
        return dhv

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        focus: torch.Tensor,
        device: str,
        filtered_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        focus = focus[..., 1]

        focus = (focus[..., None]).float()
        focus = torch.cat([focus, focus], axis=-1).to(device)
        true_grad = self.get_gradient_hv(target, device)
        pred_grad = self.get_gradient_hv(input, device)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.gamma_f = gamma_f
        self.smooth = smooth
        self.num_classes = 2

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        filtered_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError("Wrong number of channels in input")
        if len(target.shape) != len(input.shape):
            target = F.one_hot(target, num_classes=self.num_classes)

        target = target.permute(0, 2, 3, 1)
        input_soft = torch.softmax(input, dim=-1)

        if filtered_mask is not None:
            mask_4d = filtered_mask.permute(0,2,3,1).expand(-1, -1, -1, self.num_classes)
            input_soft = input_soft * mask_4d
            target = target * mask_4d

        input_flat = input_soft.view(-1)
        target_flat = target.view(-1)

        tp = (input_flat * target_flat).sum()
        fp = ((1 - target_flat) * input_flat).sum()
        fn = (target_flat * (1 - input_flat)).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        focal_tversky = (1 - tversky) ** self.gamma_f

        return focal_tversky


class MCFocalTverskyLoss(FocalTverskyLoss):
    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 0.000001,
        num_classes: int = 2,
        class_weights: List[int] = None,
    ) -> None:
        super().__init__(alpha_t, beta_t, gamma_f, smooth)
        self.num_classes = num_classes
        if class_weights is None:
            self.class_weights = [1 for i in range(self.num_classes)]
        else:
            assert len(class_weights) == self.num_classes, "Please provide matching weights"
            self.class_weights = class_weights
        self.class_weights = torch.Tensor(self.class_weights)
        
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        filtered_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError("Mismatch in num_classes")
        if len(target.shape) != len(input.shape):
            target = F.one_hot(target, num_classes=self.num_classes)

        target = target.permute(0, 2, 3, 1)
        input_soft = torch.softmax(input, dim=-1)

        if filtered_mask is not None:
            mask_4d = filtered_mask.permute(0,2,3,1).expand(-1, -1, -1, self.num_classes)
            input_soft = input_soft * mask_4d
            target = target * mask_4d

        input_soft = torch.permute(input_soft, (3, 1, 2, 0))
        target = torch.permute(target, (3, 1, 2, 0))
        input_soft = torch.flatten(input_soft, start_dim=1)
        target = torch.flatten(target, start_dim=1)

        tp = torch.sum(input_soft * target, 1)
        fp = torch.sum((1 - target) * input_soft, 1)
        fn = torch.sum(target * (1 - input_soft), 1)

        tversky = (tp + self.smooth) / (tp + self.alpha_t * fn + self.beta_t * fp + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma_f

        self.class_weights = self.class_weights.to(focal_tversky.device)
        return torch.sum(self.class_weights * focal_tversky)

class WeightedBaseLoss(nn.Module):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        class_weights: torch.Tensor = None,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.apply_sd = apply_sd
        self.apply_ls = apply_ls
        self.apply_svls = apply_svls
        self.apply_mask = apply_mask
        self.class_weights = class_weights
        self.edge_weight = edge_weight

    def apply_spectral_decouple(self, loss_matrix: torch.Tensor, yhat: torch.Tensor, lam: float = 0.01) -> torch.Tensor:
        return loss_matrix + (lam / 2) * (yhat**2).mean(axis=1)

    def apply_ls_to_target(self, target: torch.Tensor, num_classes: int, label_smoothing: float = 0.1) -> torch.Tensor:
        return target * (1 - label_smoothing) + label_smoothing / num_classes

    def apply_svls_to_target(self, target: torch.Tensor, num_classes: int, kernel_size: int = 5, sigma: int = 3, **kwargs) -> torch.Tensor:
        my, mx = kernel_size // 2, kernel_size // 2
        gaussian_kernel = gaussian_kernel2d(
            kernel_size, sigma, num_classes, device=target.device
        )
        neighborsum = (1 - gaussian_kernel[..., my, mx]) + 1e-16
        gaussian_kernel = gaussian_kernel.clone()
        gaussian_kernel[..., my, mx] = neighborsum
        svls_kernel = gaussian_kernel / neighborsum[0]
        return filter2D(target.float(), svls_kernel) / svls_kernel[0].sum()

    def apply_class_weights(self, loss_matrix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight_mat = self.class_weights[target.long()].to(target.device)
        loss = loss_matrix * weight_mat
        return loss

    def apply_edge_weights(self, loss_matrix: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        return loss_matrix * self.edge_weight**weight_map

    def apply_mask_weight(self, loss_matrix: torch.Tensor, mask: torch.Tensor, norm: bool = True) -> torch.Tensor:
        loss_matrix *= mask
        if norm:
            norm_mask = torch.mean(mask.float()) + 1e-7
            loss_matrix /= norm_mask
        return loss_matrix

    def extra_repr(self) -> str:
        s = "apply_sd={apply_sd}, apply_ls={apply_ls}, apply_svls={apply_svls}, apply_mask={apply_mask}, class_weights={class_weights}, edge_weight={edge_weight}"
        return s.format(**self.__dict__)

    def apply_filtered_mask(self, loss_matrix: torch.Tensor, filtered_mask: torch.Tensor, broadcast_to_channel: bool = False, n_channels: int = 1) -> torch.Tensor:
        if filtered_mask is None:
            return loss_matrix
        if broadcast_to_channel:
            mask_bc = filtered_mask.expand(-1, n_channels, -1, -1)
            return loss_matrix * mask_bc
        else:
            if len(loss_matrix.shape) == 3:  
                return loss_matrix * filtered_mask.squeeze(1).float()
            elif len(loss_matrix.shape) == 4:
                return loss_matrix * filtered_mask.float()
            else:
                raise ValueError("Unexpected shape for loss_matrix")
            

class MAEWeighted(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 1e-4,
        apply_sd: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        super().__init__(apply_sd, False, False, apply_mask, False, edge_weight)
        self.alpha = alpha
        self.eps = 1e-7

    def forward(self, input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None, mask: torch.Tensor = None, filtered_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        yhat = input
        n_classes = yhat.shape[1]
        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(n_classes, dim=1)
        mae_loss = torch.mean(torch.abs(target - yhat), dim=1)
        if filtered_mask is not None:
            mask_3d = filtered_mask.squeeze(1).float()
            mae_loss = mae_loss * mask_3d
            valid_count = mask_3d.sum()
            mae_loss_val = mae_loss.sum() / (valid_count + 1e-8)
        else:
            mae_loss_val = mae_loss.mean()
        if self.apply_mask and mask is not None:
            mae_loss_val_map = self.apply_mask_weight(mae_loss, mask, norm=True)
            mae_loss_val = mae_loss_val_map.mean()
            if self.alpha > 0:
                reg = torch.mean(((1 - mask).unsqueeze(1)) * torch.abs(yhat), axis=1)
                mae_loss_val += self.alpha * reg.mean()
        if self.apply_sd:
            mae_loss_map = torch.mean(torch.abs(target - yhat), dim=1)
            if filtered_mask is not None:
                mae_loss_map = mae_loss_map * filtered_mask.squeeze(1).float()
            mae_loss_map = self.apply_spectral_decouple(mae_loss_map, yhat)
            mae_loss_val = mae_loss_map.mean()
        if self.edge_weight is not None and target_weight is not None:
            mae_map = torch.mean(torch.abs(target - yhat), dim=1)
            if filtered_mask is not None:
                mae_map = mae_map * filtered_mask.squeeze(1).float()
            mae_map = self.apply_edge_weights(mae_map, target_weight)
            mae_loss_val = mae_map.mean()
        return mae_loss_val

class MSEWeighted(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight)

    @staticmethod
    def tensor_one_hot(type_map: torch.Tensor, n_classes: int) -> torch.Tensor:
        if not type_map.dtype == torch.int64:
            raise TypeError(f"Input `type_map` should have dtype: torch.int64. Got: {type_map.dtype}.")
        one_hot = torch.zeros(
            type_map.shape[0],
            n_classes,
            *type_map.shape[1:],
            device=type_map.device,
            dtype=type_map.dtype,
        )
        return one_hot.scatter_(dim=1, index=type_map.unsqueeze(1), value=1.0) + 1e-7

    def forward(self, input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None, mask: torch.Tensor = None, filtered_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        yhat = input
        num_classes = yhat.shape[1]
        if target.size() != yhat.size():
            if target.dtype == torch.float32:
                target_one_hot = target.unsqueeze(1)
            else:
                target_one_hot = MSEWeighted.tensor_one_hot(target, num_classes)
        else:
            target_one_hot = target
        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(target_one_hot, num_classes, **kwargs)
        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(target_one_hot, num_classes, **kwargs)
        mse_map = F.mse_loss(yhat, target_one_hot, reduction="none")
        mse_map = torch.mean(mse_map, dim=1)
        if filtered_mask is not None:
            mask_3d = filtered_mask.squeeze(1).float()
            mse_map = mse_map * mask_3d
            valid_count = mask_3d.sum()
            loss_val = mse_map.sum() / (valid_count + 1e-8)
        else:
            loss_val = mse_map.mean()
        if self.apply_mask and mask is not None:
            mse_map = self.apply_mask_weight(mse_map, mask, norm=False)
            loss_val = mse_map.mean()
        if self.apply_sd:
            mse_map_sd = self.apply_spectral_decouple(mse_map, yhat)
            loss_val = mse_map_sd.mean()
        if self.class_weights is not None:
            if target.dim() == 4:
                target_idx = torch.argmax(target, dim=1)
            else:
                target_idx = target
            weighted_map = self.apply_class_weights(mse_map, target_idx)
            loss_val = weighted_map.mean()
        if self.edge_weight is not None and target_weight is not None:
            edge_map = self.apply_edge_weights(mse_map, target_weight)
            loss_val = edge_map.mean()
        return loss_val

class BCEWeighted(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight)
        self.eps = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None, mask: torch.Tensor = None, filtered_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        yhat = torch.clamp(input, self.eps, 1.0 - self.eps)
        num_classes = yhat.shape[1]
        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(num_classes, dim=1)
        if self.apply_svls:
            target = self.apply_svls_to_target(target, num_classes, **kwargs)
        if self.apply_ls:
            target = self.apply_ls_to_target(target, num_classes, **kwargs)
        bce_map = F.binary_cross_entropy_with_logits(yhat, target.float(), reduction="none")
        bce_map = torch.mean(bce_map, dim=1)
        if filtered_mask is not None:
            mask_3d = filtered_mask.squeeze(1).float()
            bce_map = bce_map * mask_3d
            valid_count = mask_3d.sum()
            loss_val = bce_map.sum() / (valid_count + 1e-8)
        else:
            loss_val = bce_map.mean()
        if self.apply_mask and mask is not None:
            bce_map = self.apply_mask_weight(bce_map, mask, norm=False)
            loss_val = bce_map.mean()
        if self.apply_sd:
            bce_map_sd = self.apply_spectral_decouple(bce_map, yhat)
            loss_val = bce_map_sd.mean()
        if self.class_weights is not None:
            target_idx = torch.argmax(target, dim=1)
            bce_map_w = self.apply_class_weights(bce_map, target_idx)
            loss_val = bce_map_w.mean()
        if self.edge_weight is not None and target_weight is not None:
            bce_map_e = self.apply_edge_weights(bce_map, target_weight)
            loss_val = bce_map_e.mean()
        return loss_val


class CEWeighted(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.eps = 1e-8

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        filtered_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        yhat = F.softmax(input, dim=1) + self.eps
        num_classes = yhat.shape[1]

        if len(target.shape) != len(yhat.shape) or target.shape[1] != num_classes:
            target_one_hot = MSEWeighted.tensor_one_hot(target, num_classes)
            target_idx = target
        else:
            target_one_hot = target
            target_idx = torch.argmax(target, dim=1)

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(target_one_hot, num_classes, **kwargs)
        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(target_one_hot, num_classes, **kwargs)

        ce_map = -torch.sum(target_one_hot * torch.log(yhat), dim=1)

        if filtered_mask is not None:
            mask_3d = filtered_mask.squeeze(1).float()
            ce_map = ce_map * mask_3d
            valid_count = mask_3d.sum()
            loss_val = ce_map.sum() / (valid_count + 1e-8)
        else:
            loss_val = ce_map.mean()

        if self.apply_mask and mask is not None:
            ce_map = self.apply_mask_weight(ce_map, mask, norm=False)
            loss_val = ce_map.mean()

        if self.apply_sd:
            ce_map_sd = self.apply_spectral_decouple(ce_map, yhat)
            loss_val = ce_map_sd.mean()

        if self.class_weights is not None:
            ce_map_w = self.apply_class_weights(ce_map, target_idx)
            loss_val = ce_map_w.mean()

        if self.edge_weight is not None and target_weight is not None:
            ce_map_e = self.apply_edge_weights(ce_map, target_weight)
            loss_val = ce_map_e.mean()

        return loss_val

class L1LossWeighted(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        filtered_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        l1loss = F.l1_loss(input, target, reduction='none')
        l1loss = torch.mean(l1loss, dim=1)
        if target_weight is not None:
            l1loss = torch.mean(target_weight * l1loss)
        else:
            l1loss = torch.mean(l1loss)
        return l1loss

def retrieve_loss_fn(loss_name: dict, **kwargs) -> _Loss:
    loss_fn = LOSS_DICT[loss_name]
    loss_fn = loss_fn(**kwargs)
    return loss_fn

LOSS_DICT = {
    "xentropy_loss": XentropyLoss,
    "dice_loss": DiceLoss,
    "mse_loss_maps": MSELossMaps,
    "msge_loss_maps": MSGELossMaps,
    "FocalTverskyLoss": FocalTverskyLoss,
    "MCFocalTverskyLoss": MCFocalTverskyLoss,
    "myMSELoss": DensityMSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CTCLoss": nn.CTCLoss,
    "NLLLoss": nn.NLLLoss,
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "GaussianNLLLoss": nn.GaussianNLLLoss,
    "KLDivLoss": nn.KLDivLoss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "MarginRankingLoss": nn.MarginRankingLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "HuberLoss": nn.HuberLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "SoftMarginLoss": nn.SoftMarginLoss,
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    "MAEWeighted": MAEWeighted,
    "MSEWeighted": MSEWeighted,
    "BCEWeighted": BCEWeighted,
    "CEWeighted": CEWeighted,
    "L1LossWeighted": L1LossWeighted,
}
