import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data
import torchvision as tv

import pytorch_lightning as pl

from selectivesearch import selective_search

from .dataloader import OxfordIIITPet


class SimplifyAnnotation(object):
    def __init__(self):
        ...

    def __call__(self, sample):
        objects = sample["annotation"]["object"]
        bboxs, labels = [], []
        for obj in objects:
            bbox = obj["bndbox"]
            bboxs.append(
                (
                    int(bbox["xmin"]),
                    int(bbox["ymin"]),
                    int(bbox["xmax"]),
                    int(bbox["ymax"]),
                )
            )
            labels.append(obj["name"])
        return bboxs, labels


def x1y1x2y2_to_xywh(bbox, xy_location="center"):
    x1, y1, x2, y2 = bbox
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    if xy_location == "center":
        x, y = x + w / 2, y + h / 2
    return x, y, w, h


def xywh_to_x1y1x2y2(bbox, xy_location="center"):
    x, y, w, h = bbox
    if xy_location == "top_left":
        x1, y1 = x, y
    elif xy_location == "center":
        x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x1 + w, y1 + h
    return x1, y1, x2, y2


def get_candidates(img):
    candidates = []
    _, regions = selective_search(img, scale=100.0, sigma=0.8, min_size=20)
    img_w, img_h = img.shape[:2]
    for region in regions:
        x, y, w, h = region["rect"]
        if w < 20 or h < 20:
            continue
        if w > img_w or h > img_h:
            continue
        candidates.append(list(region["rect"]))
    return candidates


def iou(bbox1, bbox2, epsilon=1e-5):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    iou = intersection_area / bbox1_area + bbox2_area - intersection_area + epsilon
    return iou


def prep_dataset(dataset, limit):
    df = pd.DataFrame(
        columns=["gt_bbox", "label", "roi", "iou", "img_path", "img_w", "img_h"]
    )

    for i in range(len(dataset)):

        if limit is not None and i >= limit:
            break

        candidates = get_candidates(dataset[i][0].permute(1, 2, 0).numpy())

        # Convert candidates to x1 y1 x2 y2 format
        candidates = [xywh_to_x1y1x2y2(c, "top_left") for c in candidates]

        # IOU of all candidates with all ground truths
        ious = np.array([[iou(c, g) for g in dataset[i][1][0]] for c in candidates])

        for idx, candidate in enumerate(candidates):
            cx1, cy1, cx2, cy2 = candidate
            candidate_ious = ious[idx]

            # Find which ground truth has the highest IOU with each candidate
            best_iou_idx = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_iou_idx]
            # Best ground truth bbox for this candidate,
            # i.e, which ground truth is used to calculate offsets
            best_bbox = dataset[i][1][0][best_iou_idx]

            if best_iou > 0.3:
                # Give label
                label = dataset[i][1][1][best_iou_idx]
            else:
                # Give background label
                label = "background"

            # Calculate ROIs
            img_w, img_h = dataset[i][0].shape[1:]

            # Calculate offsets
            gt_x, gt_y, gt_w, gt_h = x1y1x2y2_to_xywh(best_bbox)
            cx, cy, cw, ch = x1y1x2y2_to_xywh(candidate)
            tx = (gt_x - cx) / cw
            ty = (gt_y - cy) / ch
            tw = np.log(gt_w / cw)
            th = np.log(gt_h / ch)

            # Add to dataframe
            row = pd.Series(
                {
                    "gt_bbox": np.array(best_bbox),
                    "label": label,
                    "roi": np.array(candidate),
                    "iou": best_iou,
                    "img_path": dataset.images[i],
                    "img_w": img_w,
                    "img_h": img_h,
                    "offsets": np.array([tx, ty, tw, th]),
                }
            )

            # Concatenate to dataframe
            df = pd.concat([df, row.to_frame().T], ignore_index=True)

    return df


class RCNNDataset(data.Dataset):
    def __init__(self, root, padding=5, limit=None):
        super().__init__()

        self.padding = padding

        img_transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
            ]
        )

        ann_transform = tv.transforms.Compose(
            [
                SimplifyAnnotation(),
            ]
        )

        self.dataset = OxfordIIITPet(
            root,
            img_transform=img_transform,
            ann_transform=ann_transform,
        )

        self.df = prep_dataset(self.dataset, limit)

        # Convert labels to integers
        labels = self.df["label"].unique()
        self.label_to_int = {l: i for i, l in enumerate(labels)}
        self.int_to_label = {i: l for i, l in enumerate(labels)}
        self.df["label"] = self.df["label"].apply(lambda x: self.label_to_int[x])

        self.df = (
            self.df.sort_values(by=["label"]).sample(frac=1).reset_index(drop=True)
        )

        # Save dataframe to disk
        # self.df.to_csv(root + "/rcnn_df.csv")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s = self.df.iloc[idx]

        # Open and create image crop with padding
        img = Image.open(s.img_path)
        x1, y1, x2, y2 = s.roi
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(s.img_w, x2 + self.padding)
        y2 = min(s.img_h, y2 + self.padding)
        img = img.crop((x1, y1, x2, y2))

        # Warp image to 224x224
        img = tv.transforms.Resize((224, 224))(img)

        # Convert to tensor
        img = tv.transforms.ToTensor()(img)

        offset = s.offsets
        cls = s.label

        return img, offset, cls


class RCNN(pl.LightningModule):
    def __init__(self, num_classes, lr=0.001):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr

        weights = tv.models.VGG16_BN_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.backbone = tv.models.vgg16_bn(weights=tv.models.VGG16_BN_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(self.num_classes),
        )
        self.bbox_regressor = nn.Sequential(
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(4),
        )

        self.regression_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x), self.bbox_regressor(x)

    def training_step(self, batch, batch_idx):
        imgs, offsets, labels = batch
        pred_labels, pred_offsets = self(imgs)
        regression_loss = self.regression_loss_fn(pred_offsets, offsets)
        classification_loss = self.classification_loss_fn(pred_labels, labels)
        return regression_loss + classification_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
