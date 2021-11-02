# encoding: utf-8
# python3 tools/train.py -f exps/helix/FMV/yolox_x_ablation_fmv.py --devices 8 --batch-size 64 --fp16 --occupy --experiment-name fmv_exp_02_crowdhuman --ckpt pretrained/bytetrack_x_mot17.pth.tar mosaic_dataset True motion True

from loguru import logger
import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.depth = 1.33  # X
        self.width = 1.25  # X
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.motion = False

        # When motion is False, COCO for train
        # self.train_ann = "training_coco_crowds_61_07292021_intids.json"
        # self.train_ann = "train_08062021_81453_intids.json"
        # COCO for validation
        self.val_ann = "val_08302021_491_intids.json"
        self.downsample_mod = 30

        self.input_size = (736, 1280)
        self.test_size = (736, 1280)
        self.random_size = (20, 30)  # 18,32?
        self.max_epoch = 20
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.1
        self.nmsthre = 0.7

        self.no_aug_epochs = 3
        self.basic_lr_per_img = 0.001 / 64.0

        self.warmup_epochs = 1
        self.data_num_workers = 8
        self.distort = True
        self.mirror = True
        self.scale = (0.5, 1.5)
        self.scale_bbox_height = 1.0
        self.scale_bbox_width = 1.0
        self.max_labels = 300
        self.max_labels_mosaic = 500
        self.debug_limit = None
        self.save_image_examples = False
        self.mosaic_dataset = False
        self.enable_mixup = False
        self.add_graph = False

        self.min_lr_ratio = 0.05

        self.tsm = False



        # Get training multiscale input sizes for logging
        size_factor = self.input_size[1] * 1.0 / self.input_size[0]
        self.multiscale_training_sizes = []
        for s in range(self.random_size[0], self.random_size[1]+1):
            self.multiscale_training_sizes.append((int(32 * s), 32 * int(s * size_factor)))

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            MotionDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        if self.motion:
            logger.info('Using MotionDataset for training.')
            dataset = MotionDataset(
                root_path=os.path.join(get_yolox_datadir(), "FMV", "motion"),
                anno_file="anno.json",
                split_file="splits.json",
                is_train=True,
                tsm=self.tsm,
                img_size=self.input_size,
                preproc=TrainTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_labels=self.max_labels,
                    distort=self.distort,
                    mirror=self.mirror,
                    scale_bbox_height=self.scale_bbox_height,
                    scale_bbox_width=self.scale_bbox_width,
                ),
                debug_limit=self.debug_limit,
                save_image_examples=self.save_image_examples,
            )

        else:
            logger.info('Using MOTDataset for training.')
            dataset = MOTDataset(
                data_dir=os.path.join(get_yolox_datadir(), "FMV"),
                json_file=self.train_ann,
                name='fmv_sequence',
                img_size=self.input_size,
                preproc=TrainTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_labels=self.max_labels,
                    distort=self.distort,
                    mirror=self.mirror,
                    scale_bbox_height=self.scale_bbox_height,
                    scale_bbox_width=self.scale_bbox_width,
                ),
            )

        if self.mosaic_dataset:
            logger.info('Wrapping training dataset with MosaicDetection.')
            dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_labels=self.max_labels_mosaic,
                    distort=self.distort,
                    mirror=self.mirror,
                    scale_bbox_height=self.scale_bbox_height,
                    scale_bbox_width=self.scale_bbox_width,
                ),
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                enable_mixup=self.enable_mixup,
                save_image_examples=self.save_image_examples,
            )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "FMV"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='fmv_sequence',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            downsample_mod=self.downsample_mod,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
