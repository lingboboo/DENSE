import sys  
import copy  
import datetime
import inspect 
import os  
import shutil 
import yaml 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
parentdir = os.path.dirname(currentdir) 
sys.path.insert(0, parentdir) 
import uuid 
from pathlib import Path 
from typing import Callable, Tuple, Union 
import albumentations as A 
import torch  
import torch.nn as nn  
import wandb  
from torch.optim import Optimizer  
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    SequentialLR,
    _LRScheduler,
) 
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    Subset,
    WeightedRandomSampler,
) 
from torchinfo import summary 
from wandb.sdk.lib.runid import generate_id 

from base_ml.base_early_stopping import EarlyStopping 
from base_ml.base_experiment import BaseExperiment 
from base_loss import retrieve_loss_fn  
from base_ml.base_trainer import BaseTrainer  
from nuclei_segmentation.datasets.base_nuclei import nucleiDataset 
from nuclei_segmentation.datasets.dataset_coordinator import select_dataset  
from nuclei_segmentation.trainer.trainer_Dense import DenseTrainer 
from models.segmentation.nuclei_segmentation.Dense import Dense
from utils.tools import close_logger 

class ExperimentDensePanNuke(BaseExperiment):
    def __init__(self, default_conf: dict, checkpoint=None) -> None:
        super().__init__(default_conf, checkpoint)  
        self.load_dataset_setup(dataset_path=self.default_conf["data"]["dataset_path"])  

    def run_experiment(self) -> tuple[Path, dict, nn.Module, dict]:
        """Main Experiment Code"""
        self.close_remaining_logger() 

        self.run_conf = copy.deepcopy(self.default_conf)  
        self.run_conf["dataset_config"] = self.dataset_config  
        self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')}_{self.run_conf['logging']['log_comment']}"  

        wandb_run_id = generate_id()  
        resume = None 
        if self.checkpoint is not None:
            wandb_run_id = self.checkpoint["wandb_id"]  
            resume = "must"  
            self.run_name = self.checkpoint["run_name"]

        # initialize wandb
        run = wandb.init(
            project=self.run_conf["logging"]["project"],  
            tags=self.run_conf["logging"].get("tags", []), 
            name=self.run_name,  
            notes=self.run_conf["logging"]["notes"],
            dir=self.run_conf["logging"]["wandb_dir"], 
            mode=self.run_conf["logging"]["mode"].lower(), 
            group=self.run_conf["logging"].get("group", str(uuid.uuid4())), 
            allow_val_change=True, 
            id=wandb_run_id, 
            resume=resume, 
            settings=wandb.Settings(start_method="fork"),  
        )  

        self.run_conf["logging"]["run_id"] = run.id  
        self.run_conf["logging"]["wandb_file"] = run.id

        if self.run_conf["run_sweep"] is True: 
            self.run_conf["logging"]["sweep_id"] = run.sweep_id 
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"])
                / f"sweep_{run.sweep_id}"
                / f"{self.run_name}_{self.run_conf['logging']['run_id']}"
            )  
            self.overwrite_sweep_values(self.run_conf, run.config) 
        else:
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"]) / self.run_name
            ) 
        # update wandb
        wandb.config.update(
            self.run_conf, allow_val_change=True
        ) 

        self.create_output_dir(self.run_conf["logging"]["log_dir"])  
        self.logger = self.instantiate_logger() 
        self.logger.info("Instantiated Logger. WandB init and config update finished.")  
        self.logger.info(f"Run ist stored here: {self.run_conf['logging']['log_dir']}")  
        self.store_config() 
        self.logger.info(
            f"Cuda devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )  
        device = f"cuda:{self.run_conf['gpu']}"  
        self.logger.info(f"Using GPU: {device}")  
        self.logger.info(f"Using device: {device}") 

        loss_fn_dict = self.get_loss_fn(self.run_conf.get("loss", {}))  
        self.logger.info("Loss functions:")  
        self.logger.info(loss_fn_dict)  

        # model
        model = self.get_train_model(
            pretrained_encoder=self.run_conf["model"].get("pretrained_encoder", None), 
            pretrained_model=self.run_conf["model"].get("pretrained", None),  
            backbone_type=self.run_conf["model"].get("backbone", "default"),  
            shared_decoders=self.run_conf["model"].get("shared_decoders", False),  
            regression_loss=self.run_conf["model"].get("regression_loss", False),  
            den_loss=self.run_conf["model"].get("den_loss", False),             
        ) 
        model.to(device) 

        optimizer = self.get_optimizer(
            model,
            self.run_conf["training"]["optimizer"], 
            self.run_conf["training"]["optimizer_hyperparameter"],  
        )  

        scheduler = self.get_scheduler(
            optimizer=optimizer,  
            scheduler_type=self.run_conf["training"]["scheduler"]["scheduler_type"],  
        )  

        early_stopping = None  
        if "early_stopping_patience" in self.run_conf["training"]: 
            if self.run_conf["training"]["early_stopping_patience"] is not None:
                early_stopping = EarlyStopping(
                    patience=self.run_conf["training"]["early_stopping_patience"], 
                    strategy="maximize", 
                ) 

        train_transforms, val_transforms = self.get_transforms(
            self.run_conf["transformations"],  
            input_shape=self.run_conf["data"].get("input_shape", 256),  
        )  

        train_dataset, val_dataset = self.get_datasets(
            train_transforms=train_transforms, 
            val_transforms=val_transforms,  
        )  

        training_sampler = self.get_sampler(
            train_dataset=train_dataset, 
            strategy=self.run_conf["training"].get("sampling_strategy", "random"), 
            gamma=self.run_conf["training"].get("sampling_gamma", 1), 
        )  

        train_dataloader = DataLoader(
            train_dataset,  
            batch_size=self.run_conf["training"]["batch_size"], 
            sampler=training_sampler, 
            num_workers=16,  
            pin_memory=False,  
            worker_init_fn=self.seed_worker,  
        )  

        val_dataloader = DataLoader(
            val_dataset,  
            batch_size=16,  
            num_workers=16, 
            pin_memory=True,  
            worker_init_fn=self.seed_worker,  
        )  

        self.logger.info("Instantiate Trainer") 
        trainer_fn = self.get_trainer()
        trainer = trainer_fn(
            model=model, 
            loss_fn_dict=loss_fn_dict, 
            optimizer=optimizer,  
            scheduler=scheduler,  
            device=device,  
            logger=self.logger,  
            logdir=self.run_conf["logging"]["log_dir"],  
            dataset_config=self.dataset_config,  
            early_stopping=early_stopping, 
            experiment_config=self.run_conf,
            log_images=self.run_conf["logging"].get("log_images", False), 
            magnification=self.run_conf["data"].get("magnification", 40),  
            mixed_precision=self.run_conf["training"].get("mixed_precision", False), 
        ) 

        if self.checkpoint is not None:  
            self.logger.info("Checkpoint was provided. Restore ...")  
            trainer.resume_checkpoint(self.checkpoint) 

        self.logger.info("Calling Trainer Fit")  
        trainer.fit(
            epochs=self.run_conf["training"]["epochs"],
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,  
            metric_init=self.get_wandb_init_dict(),  
            unfreeze_epoch=self.run_conf["training"]["unfreeze_epoch"], 
            eval_every=self.run_conf["training"].get("eval_every", 1),  
        )  

        checkpoint_dir = Path(self.run_conf["logging"]["log_dir"]) / "checkpoints"  
        if not (checkpoint_dir / "model_best.pth").is_file():  
            shutil.copy(
                checkpoint_dir / "latest_checkpoint.pth",  
                checkpoint_dir / "model_best.pth", 
            )  

        self.logger.info(f"Finished run {run.id}") 
        close_logger(self.logger) 

        return self.run_conf["logging"]["log_dir"]  

    def load_dataset_setup(self, dataset_path: Union[Path, str]) -> None:

        dataset_config_path = Path(dataset_path) / "dataset_config.yaml"  
        with open(dataset_config_path, "r") as dataset_config_file:
            yaml_config = yaml.safe_load(dataset_config_file) 
            self.dataset_config = dict(yaml_config)  

    def get_loss_fn(self, loss_fn_settings: dict) -> dict:

        loss_fn_dict = {}  
        if "nuclei_binary_map" in loss_fn_settings.keys():  
            loss_fn_dict["nuclei_binary_map"] = {} 
            for loss_name, loss_sett in loss_fn_settings["nuclei_binary_map"].items():
                parameters = loss_sett.get("args", {}) 
                loss_fn_dict["nuclei_binary_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters), 
                    "weight": loss_sett["weight"],
                } 
        else:
            loss_fn_dict["nuclei_binary_map"] = {
                "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
                "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
            }  

        if "hv_map" in loss_fn_settings.keys(): 
            loss_fn_dict["hv_map"] = {} 
            for loss_name, loss_sett in loss_fn_settings["hv_map"].items(): 
                parameters = loss_sett.get("args", {}) 
                loss_fn_dict["hv_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters), 
                    "weight": loss_sett["weight"],  
                } 
        else:
            loss_fn_dict["hv_map"] = {
                "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 1},
                "msge": {"loss_fn": retrieve_loss_fn("msge_loss_maps"), "weight": 1},
            } 


        if "den_map" in loss_fn_settings.keys(): 
            loss_fn_dict["den_map"] = {} 
            for loss_name, loss_sett in loss_fn_settings["den_map"].items(): 
                parameters = loss_sett.get("args", {})  
                loss_fn_dict["den_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters), 
                    "weight": loss_sett["weight"],  
                } 
        elif "den_loss" in self.run_conf["model"].keys(): 
            loss_fn_dict["den_map"] = {
                "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 1},
            } 

        if "regression_loss" in loss_fn_settings.keys(): 
            loss_fn_dict["regression_map"] = {} 
            for loss_name, loss_sett in loss_fn_settings["regression_loss"].items():  
                loss_fn_dict["regression_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),  
                    "weight": loss_sett["weight"],
                } 
        elif "regression_loss" in self.run_conf["model"].keys():  
            loss_fn_dict["regression_map"] = {
                "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 1},
            }  

        return loss_fn_dict 

    def get_scheduler(self, scheduler_type: str, optimizer: Optimizer) -> _LRScheduler:

        implemented_schedulers = ["constant", "exponential", "cosine"] 
        if scheduler_type.lower() not in implemented_schedulers:  
            self.logger.warning(
                f"Unknown Scheduler - No scheduler from the list {implemented_schedulers} selected. Using default scheduling."
            ) 
        if scheduler_type.lower() == "constant": 
            scheduler = SequentialLR(
                optimizer=optimizer,  
                schedulers=[
                    ConstantLR(optimizer, factor=1, total_iters=25),  
                    ConstantLR(optimizer, factor=0.1, total_iters=25), 
                    ConstantLR(optimizer, factor=1, total_iters=25), 
                    ConstantLR(optimizer, factor=0.1, total_iters=1000), 
                ],
                milestones=[24, 49, 74],  
            ) 
        elif scheduler_type.lower() == "exponential": 
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.run_conf["training"]["scheduler"].get("gamma", 0.95), 
            ) 
        elif scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.run_conf["training"]["epochs"],  
                eta_min=self.run_conf["training"]["scheduler"].get("eta_min", 1e-5),  
            )  
        else:
            scheduler = super().get_scheduler(optimizer)  
        return scheduler 

    def get_datasets(
        self,
        train_transforms: Callable = None,
        val_transforms: Callable = None,
    ) -> Tuple[Dataset, Dataset]:
        if (
            "val_split" in self.run_conf["data"]
            and "val_folds" in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_splits or val_folds in configuration file, not both."
            ) 
        if (
            "val_split" not in self.run_conf["data"]
            and "val_folds" not in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_split or val_folds in configuration file, one is necessary."
            )  
        if (
            "val_split" not in self.run_conf["data"]
            and "val_folds" not in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_split or val_fold in configuration file, one is necessary."
            ) 
        if "regression_loss" in self.run_conf["model"].keys():
            self.run_conf["data"]["regression_loss"] = True  
        if "den_loss" in self.run_conf["model"].keys():
            self.run_conf["data"]["den_loss"] = True 

        full_dataset = select_dataset(
            dataset_name="pannuke",  
            split="train",  
            dataset_config=self.run_conf["data"], 
            transforms=train_transforms,
        )
        if "val_split" in self.run_conf["data"]: 
            generator_split = torch.Generator().manual_seed(
                self.default_conf["random_seed"]
            )  
            val_splits = float(self.run_conf["data"]["val_split"])
            train_len = int(len(full_dataset) * (1 - val_splits)) 
            val_len = len(full_dataset) - train_len 
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset,
                lengths=[train_len, val_len],  
                generator=generator_split, 
            ) 
            val_dataset.dataset = copy.deepcopy(full_dataset)  
            val_dataset.dataset.set_transforms(val_transforms)  
        else:
            train_dataset = full_dataset 
            val_dataset = select_dataset(
                dataset_name="pannuke", 
                split="validation", 
                dataset_config=self.run_conf["data"], 
                transforms=val_transforms, 
            )  
        return train_dataset, val_dataset  

    def get_train_model(
        self,
        pretrained_encoder: Union[Path, str] = None,
        pretrained_model: Union[Path, str] = None,
        backbone_type: str = "default",
        shared_decoders: bool = False,
        regression_loss: bool = False,
        den_loss: bool = False,
        **kwargs,
    ) -> Dense:
   
        # reseed needed, due to subprocess seeding compatibility
        self.seed_run(self.default_conf["random_seed"]) 

        # check for backbones
        implemented_backbones = ["default"]  
        if backbone_type.lower() not in implemented_backbones: 
            raise NotImplementedError(
                f"Unknown Backbone Type - Currently supported are: {implemented_backbones}"
            )  
        if backbone_type.lower() == "default": 
            model_class = Dense  
            model = model_class(
                embed_dim=self.run_conf["model"]["embed_dim"], 
                input_channels=self.run_conf["model"].get("input_channels", 3),  
                depth=self.run_conf["model"]["depth"], 
                num_heads=self.run_conf["model"]["num_heads"],  
                extract_layers=self.run_conf["model"]["extract_layers"],  
                drop_rate=self.run_conf["training"].get("drop_rate", 0),  
                attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),  
                drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0), 
                regression_loss=regression_loss,  
                den_loss=den_loss, 
            ) 

            if pretrained_model is not None:  
                self.logger.info(
                    f"Loading pretrained Dense model from path: {pretrained_model}"
                ) 
                Dense_pretrained = torch.load(pretrained_model) 
                self.logger.info(model.load_state_dict(Dense_pretrained, strict=True)) 
                self.logger.info("Loaded Dense model") 

        self.logger.info(f"\nModel: {model}")  
        model = model.to("cpu") 
        self.logger.info(
            f"\n{summary(model, input_size=(1, 3, 256, 256), device='cpu')}"
        ) 

        return model  

    def get_wandb_init_dict(self) -> dict:
        pass 
    
    def get_transforms(self, transform_settings: dict, input_shape: int = 256) -> Tuple[Callable, Callable]:
        transform_list = []
        transform_settings = {k.lower(): v for k, v in transform_settings.items()}
        if "RandomRotate90".lower() in transform_settings:
            p = transform_settings["randomrotate90"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.RandomRotate90(p=p))
        if "HorizontalFlip".lower() in transform_settings.keys():
            p = transform_settings["horizontalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.HorizontalFlip(p=p))
        if "VerticalFlip".lower() in transform_settings:
            p = transform_settings["verticalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.VerticalFlip(p=p))
        if "Downscale".lower() in transform_settings:
            p = transform_settings["downscale"]["p"]
            scale = transform_settings["downscale"]["scale"]
            if p > 0 and p <= 1:
                transform_list.append(A.Downscale(p=p, scale_max=scale, scale_min=scale))
        if "Blur".lower() in transform_settings:
            p = transform_settings["blur"]["p"]
            blur_limit = transform_settings["blur"]["blur_limit"]
            if p > 0 and p <= 1:
                transform_list.append(A.Blur(p=p, blur_limit=blur_limit))
        if "GaussNoise".lower() in transform_settings:
            p = transform_settings["gaussnoise"]["p"]
            var_limit = transform_settings["gaussnoise"]["var_limit"]
            if p > 0 and p <= 1:
                transform_list.append(A.GaussNoise(p=p, var_limit=var_limit))
        if "ColorJitter".lower() in transform_settings:
            p = transform_settings["colorjitter"]["p"]
            scale_setting = transform_settings["colorjitter"]["scale_setting"]
            scale_color = transform_settings["colorjitter"]["scale_color"]
            if p > 0 and p <= 1:
                transform_list.append(A.ColorJitter(
                    p=p,
                    brightness=scale_setting,
                    contrast=scale_setting,
                    saturation=scale_color,
                    hue=scale_color / 2
                ))
        if "Superpixels".lower() in transform_settings:
            p = transform_settings["superpixels"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.Superpixels(
                    p=p,
                    p_replace=0.1,
                    n_segments=200,
                    max_size=int(input_shape / 2)
                ))
        if "ZoomBlur".lower() in transform_settings:
            p = transform_settings["zoomblur"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.ZoomBlur(p=p, max_factor=1.05))
        if "RandomSizedCrop".lower() in transform_settings:
            p = transform_settings["randomsizedcrop"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.RandomSizedCrop(
                    min_max_height=(input_shape / 2, input_shape),
                    height=input_shape,
                    width=input_shape,
                    p=p
                ))
        if "ElasticTransform".lower() in transform_settings:
            p = transform_settings["elastictransform"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.ElasticTransform(p=p, sigma=25, alpha=0.5, alpha_affine=15))

        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        transform_list.append(A.Normalize(mean=mean, std=std))

        train_transforms = A.Compose(transform_list)
        val_transforms = A.Compose([A.Normalize(mean=mean, std=std)])

        return train_transforms, val_transforms

    def get_sampler(self, train_dataset: nucleiDataset, strategy: str = "random", gamma: float = 1) -> Sampler:
        if strategy.lower() == "random":
            sampling_generator = torch.Generator().manual_seed(self.default_conf["random_seed"])
            sampler = RandomSampler(train_dataset, generator=sampling_generator)
            self.logger.info("Using RandomSampler")
        else:
            print("Unknown sampling strategy ")
            if isinstance(train_dataset, Subset):
                weights = torch.Tensor([weights[i] for i in train_dataset.indices])

            sampling_generator = torch.Generator().manual_seed(self.default_conf["random_seed"])
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=sampling_generator
            )
            self.logger.info(f"Using Weighted Sampling with strategy: {strategy}")
            self.logger.info(f"Unique-Weights: {torch.unique(weights)}")

        return sampler

    def get_trainer(self) -> BaseTrainer:
        return DenseTrainer
