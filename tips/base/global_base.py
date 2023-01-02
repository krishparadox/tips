from abc import ABC
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from tips.utils.basic_utils import set_color, ModelType


class BaseGNNRecommender(nn.Module):
    def __init__(self):
        self.logger = getLogger()
        super(BaseGNNRecommender, self).__init__()

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
                super().__str__()
                + set_color("\nTrainable parameters", "blue")
                + f": {params}"
        )


class GenericGNNRecommender(BaseGNNRecommender, ABC):
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GenericGNNRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]


class GeneralGraphRecommender(GenericGNNRecommender, ABC):
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralGraphRecommender, self).__init__(config, dataset)
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)