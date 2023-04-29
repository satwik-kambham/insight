import collections
import pathlib
import os
from xml.etree.ElementTree import Element as ET_Element

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from PIL import Image

from torch.utils import data
from torchvision.datasets.utils import download_and_extract_archive


class OxfordIIITPet(data.Dataset):
    _RESOURCES = (
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        ),
        (
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            "95a8c909bbe2e81eed6a22bccdf3f68f",
        ),
    )

    def __init__(
        self,
        root,
        split="trainval",
        img_transform=None,
        ann_transform=None,
        download=True,
    ):
        super().__init__()
        self.root = root
        self._split = split if split in ["trainval", "test"] else "trainval"

        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self.images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._bboxs_folder = self._anns_folder / "xmls"

        self.img_transform = img_transform
        self.ann_transform = ann_transform

        if download:
            self._download()

        image_ids = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, *_ = line.strip().split()
                if (self.images_folder / f"{image_id}.jpg").exists() and (
                    self._bboxs_folder / f"{image_id}.xml"
                ).exists():
                    image_ids.append(image_id)

        self.images = [self.images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._bboxs = [self._bboxs_folder / f"{image_id}.xml" for image_id in image_ids]

    def _check_exists(self) -> bool:
        for folder in (self.images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self):
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(
                url, download_root=str(self._base_folder), md5=md5
            )

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")

        target = self.parse_xml(ET_parse(self._bboxs[idx]).getroot())

        if self.img_transform:
            image = self.img_transform(image)
        if self.ann_transform:
            target = self.ann_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def parse_xml(node: ET_Element):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(OxfordIIITPet.parse_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {
                node.tag: {
                    ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()
                }
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
