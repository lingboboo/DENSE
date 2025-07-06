from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from .MTGT_matrix_v3 import TG_MSA
from nuclei_segmentation.utils.post_proc_Dense import DetectionnucleiPostProcessor
from .utils import Conv2DBlock, Deconv2DBlock, ViTDense

class Dense(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
        den_loss: bool = False,
    ):
        # Initializes the Dense model with specified parameters, 
        # setting up the encoder, decoders, and task-specific branches. 
        # It configures the model architecture based on input dimensions and task requirements, such as regression and density loss.
        super().__init__()
        assert len(extract_layers) == 4, "Provide 4 layers for skip connections"

        self.patch_size = 16
        self.task_guided_attention = TG_MSA(dim=512, dim_head=int(512/8), heads=8, device=torch.device('cpu'))

        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        # Initialize the encoder with Vision Transformer (ViT) architecture
        self.encoder = ViTDense(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Determine the dimensions for skip connections and bottleneck based on embedding dimension
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # Define the decoder stages with convolutional and deconvolutional blocks
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        self.regression_loss = regression_loss
        offset_branches = 0
        if self.regression_loss:
            offset_branches = 2

         # Check if density loss is enabled and define the output branches
        self.den_loss = den_loss
        if self.den_loss:
            self.branches_output = {
                "nuclei_binary_map": 2 + offset_branches,
                "hv_map": 2,
                "den_map": 1,
            }
            self.den_map_decoder = self.create_upsampling_branch(1)
        else:
            self.branches_output = {
                "nuclei_binary_map": 2 + offset_branches,
                "hv_map": 2,
            }

        # Create the upsampling branches for different outputs
        self.nuclei_binary_map_decoder = self.create_upsampling_branch(2 + offset_branches)
        self.hv_map_decoder = self.create_upsampling_branch(2)
        if self.den_loss:
            self.adjust_z1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=768, kernel_size=1),
                nn.AdaptiveAvgPool2d((16, 16))
            )
            self.adjust_z2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=768, kernel_size=1),
                nn.AdaptiveAvgPool2d((16, 16))
            )
            self.adjust_z3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=768, kernel_size=1),
                nn.AdaptiveAvgPool2d((16, 16))
            )
            self.adjust_z4 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=768, kernel_size=1),
                nn.AdaptiveAvgPool2d((16, 16))
            )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """
        Executes the forward pass of the model, processing input data through the encoder and decoders. 
        It generates outputs for different tasks (e.g., nuclei binary map, HV map, density map) and optionally retrieves intermediate tokens.
        """
        out_dict = {}

        _, _, z = self.encoder(x)
        z0, z1, z2, z3, z4 = x, *z

        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        if self.den_loss:
            d4 = self.den_map_decoder.bottleneck_upsampler(z4)
            h4 = self.hv_map_decoder.bottleneck_upsampler(z4)
            s4 = self.nuclei_binary_map_decoder.bottleneck_upsampler(z4)

            aligned_features_d4 = self.task_guided_attention(d4, h4)
            aligned_features_h4 = self.task_guided_attention(h4, d4)
            aligned_features_s4 = self.task_guided_attention(s4, aligned_features_d4)

            d3 = self.decoder3(z3)
            d3 = self.den_map_decoder.decoder3_upsampler(torch.cat([d3, aligned_features_d4], dim=1))
            d2 = self.decoder2(z2)
            d2 = self.den_map_decoder.decoder2_upsampler(torch.cat([d2, d3], dim=1))
            d1 = self.decoder1(z1)
            d1 = self.den_map_decoder.decoder1_upsampler(torch.cat([d1, d2], dim=1))
            d0 = self.decoder0(z0)
            den_branch_output = self.den_map_decoder.decoder0_header(torch.cat([d0, d1], dim=1))
            out_dict["den_map"] = den_branch_output

            h3 = self.decoder3(z3)
            h3 = self.hv_map_decoder.decoder3_upsampler(torch.cat([h3, aligned_features_h4], dim=1))
            h2 = self.decoder2(z2)
            h2 = self.hv_map_decoder.decoder2_upsampler(torch.cat([h2, h3], dim=1))
            h1 = self.decoder1(z1)
            h1 = self.hv_map_decoder.decoder1_upsampler(torch.cat([h1, h2], dim=1))
            h0 = self.decoder0(z0)
            hv_branch_output = self.hv_map_decoder.decoder0_header(torch.cat([h0, h1], dim=1))
            out_dict["hv_map"] = hv_branch_output

            s3 = self.decoder3(z3)
            s3 = self.nuclei_binary_map_decoder.decoder3_upsampler(torch.cat([s3, aligned_features_s4], dim=1))
            s2 = self.decoder2(z2)
            s2 = self.nuclei_binary_map_decoder.decoder2_upsampler(torch.cat([s2, s3], dim=1))
            s1 = self.decoder1(z1)
            s1 = self.nuclei_binary_map_decoder.decoder1_upsampler(torch.cat([s1, s2], dim=1))
            s0 = self.decoder0(z0)
            seg_branch_output = self.nuclei_binary_map_decoder.decoder0_header(torch.cat([s0, s1], dim=1))
            out_dict["nuclei_binary_map"] = seg_branch_output
        else:
            out_dict["hv_map"], _ = self._forward_upsample(
                z0, z1, z2, z3, z4, self.hv_map_decoder
            )
            if self.regression_loss:
                nb_map, _ = self._forward_upsample(
                    z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
                )
                out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
                out_dict["regression_map"] = nb_map[:, 2:, :, :]
            else:
                out_dict["nuclei_binary_map"], _ = self._forward_upsample(
                    z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
                )

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        features = {}
        b4 = branch_decoder.bottleneck_upsampler(z4)
        features['b4'] = b4
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        features['b3'] = b3
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        features['b2'] = b2
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        features['b1'] = b1
        b0 = self.decoder0(z0)
        features['b0'] = b0
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        return branch_output, features

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )
        return decoder

    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        
        # Computes instance maps from model predictions, using post-processing to segment and identify individual instances in the output maps.
        predictions_ = predictions.copy()
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        nuclei_post_processor = DetectionnucleiPostProcessor(
            magnification=magnification, gt=False
        )
        instance_preds = []
        inst_info_list = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )

            instance_pred, inst_info_dict = nuclei_post_processor.post_process_nuclei_segmentation(pred_map)
            instance_preds.append(instance_pred)
            inst_info_list.append(inst_info_dict)

        instance_tensor = torch.Tensor(np.stack(instance_preds)).long()
        return instance_tensor, inst_info_list

    def freeze_encoder(self):
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":
                p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

@dataclass
class DataclassHVStorage:
    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    instance_map: torch.Tensor
    batch_size: int
    regression_map: torch.Tensor = None
    regression_loss: bool = False
    den_map: torch.Tensor = None
    point_map: torch.Tensor = None
    density_loss: bool = False
    mask_knn: torch.Tensor = None
    combined_mask: torch.Tensor = None
    h: int = 256
    w: int = 256

    def get_dict(self) -> dict:
        property_dict = self.__dict__
        if not self.regression_loss and "regression_map" in property_dict.keys():
            property_dict.pop("regression_map")
        if not self.density_loss and "den_map" in property_dict.keys():
            property_dict.pop("den_map")
            property_dict.pop("point_map")
            property_dict.pop("combined_mask")
        return property_dict
