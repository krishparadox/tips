import copy
import importlib
import os
import pickle
import torch
from logging import getLogger

from tips.utils.basic_utils import ensure_dir, set_color
from tips.model_builder.tips import data_args


def load_split_dataloaders(config):
    default_file = os.path.join(
        config["checkpoint_dir"],
        f'{config["dataset"]}-for-{config["model"]}-dataloader.pth',
    )
    dataloaders_save_path = config["dataloaders_save_path"] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        train_data, valid_data, test_data = dataloaders
    for arg in data_args + ["seed", "repeatable", "eval_args"]:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(
        set_color("Load split dataloaders from", "pink")
        + f": [{dataloaders_save_path}]"
    )
    return train_data, valid_data, test_data


def create_samplers(config, dataset, built_datasets):
    phases = ["train", "valid", "test"]
    train_neg_sample_args = config["train_neg_sample_args"]
    eval_neg_sample_args = config["eval_neg_sample_args"]
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args["distribution"] != "none":
        if not config["repeatable"]:
            sampler = Sampler(
                phases,
                built_datasets,
                train_neg_sample_args["distribution"],
                train_neg_sample_args["alpha"],
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                train_neg_sample_args["distribution"],
                train_neg_sample_args["alpha"],
            )
        train_sampler = sampler.set_phase("train")

    if eval_neg_sample_args["distribution"] != "none":
        if sampler is None:
            if not config["repeatable"]:
                sampler = Sampler(
                    phases,
                    built_datasets,
                    eval_neg_sample_args["distribution"],
                )
            else:
                sampler = RepeatableSampler(
                    phases,
                    dataset,
                    eval_neg_sample_args["distribution"],
                )
        else:
            sampler.set_distribution(eval_neg_sample_args["distribution"])
        valid_sampler = sampler.set_phase("valid")
        test_sampler = sampler.set_phase("test")

    return train_sampler, valid_sampler, test_sampler


def save_split_dataloaders(config, dataloaders):
    ensure_dir(config["checkpoint_dir"])
    save_path = config["checkpoint_dir"]
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
    serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(serialization_dataloaders, f)
