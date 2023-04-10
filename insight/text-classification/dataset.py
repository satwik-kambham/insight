import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import pytorch_lightning as pl

from tokenizer import CharacterTokenizer


class TREC(Dataset):
    """The Text REtrieval Conference (TREC) Question Classification dataset

    The data fields are the same among all splits.

    text (str): Text of the question.
    coarse_label (ClassLabel): Coarse class label. Possible values are:
        'ABBR' (0): Abbreviation.
        'ENTY' (1): Entity.
        'DESC' (2): Description and abstract concept.
        'HUM' (3): Human being.
        'LOC' (4): Location.
        'NUM' (5): Numeric value.
    fine_label (ClassLabel): Fine class label. Possible values are:
        ABBREVIATION:
            'ABBR:abb' (0): Abbreviation.
            'ABBR:exp' (1): Expression abbreviated.
        ENTITY:
            'ENTY:animal' (2): Animal.
            'ENTY:body' (3): Organ of body.
            'ENTY:color' (4): Color.
            'ENTY:cremat' (5): Invention, book and other creative piece.
            'ENTY:currency' (6): Currency name.
            'ENTY:dismed' (7): Disease and medicine.
            'ENTY:event' (8): Event.
            'ENTY:food' (9): Food.
            'ENTY:instru' (10): Musical instrument.
            'ENTY:lang' (11): Language.
            'ENTY:letter' (12): Letter like a-z.
            'ENTY:other' (13): Other entity.
            'ENTY:plant' (14): Plant.
            'ENTY:product' (15): Product.
            'ENTY:religion' (16): Religion.
            'ENTY:sport' (17): Sport.
            'ENTY:substance' (18): Element and substance.
            'ENTY:symbol' (19): Symbols and sign.
            'ENTY:techmeth' (20): Techniques and method.
            'ENTY:termeq' (21): Equivalent term.
            'ENTY:veh' (22): Vehicle.
            'ENTY:word' (23): Word with a special property.
        DESCRIPTION:
            'DESC:def' (24): Definition of something.
            'DESC:desc' (25): Description of something.
            'DESC:manner' (26): Manner of an action.
            'DESC:reason' (27): Reason.
        HUMAN:
            'HUM:gr' (28): Group or organization of persons
            'HUM:ind' (29): Individual.
            'HUM:title' (30): Title of a person.
            'HUM:desc' (31): Description of a person.
        LOCATION:
            'LOC:city' (32): City.
            'LOC:country' (33): Country.
            'LOC:mount' (34): Mountain.
            'LOC:other' (35): Other location.
            'LOC:state' (36): State.
        NUMERIC:
            'NUM:code' (37): Postcode or other code.
            'NUM:count' (38): Number of something.
            'NUM:date' (39): Date.
            'NUM:dist' (40): Distance, linear measure.
            'NUM:money' (41): Price.
            'NUM:ord' (42): Order, rank.
            'NUM:other' (43): Other number.
            'NUM:period' (44): Lasting time of something
            'NUM:perc' (45): Percent, fraction.
            'NUM:speed' (46): Speed.
            'NUM:temp' (47): Temperature.
            'NUM:volsize' (48): Size, area and volume.
            'NUM:weight' (49): Weight.
    """

    def __init__(self, split="train"):
        self.dataset = load_dataset("trec", split=split)
        self.tokenizer = CharacterTokenizer()

        for i in self.dataset:
            self.tokenizer.train(i['text'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        i = self.dataset[idx]
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer.encode(text)
        coarse_label = torch.as_tensor(self.dataset[idx]["coarse_label"])
        fine_label = torch.as_tensor(self.dataset[idx]["fine_label"])
        return (
            F.one_hot(torch.as_tensor(tokens), num_classes=self.tokenizer.count).float(),
            F.one_hot(coarse_label, num_classes=6).float(),
            F.one_hot(fine_label, num_classes=50).float(),
        )


class RNN_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = TREC(split="train")
        self.val_dataset = TREC(split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
