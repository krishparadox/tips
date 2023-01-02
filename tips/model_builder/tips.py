import os
import pickle
import importlib
from logging import getLogger

from tips.utils.data_utils import load_split_dataloaders, create_samplers, save_split_dataloaders
from tips.utils.basic_utils import set_color, Enum, ModelType

data_args = [
    'field_separator', 'seq_separator',
    'USER_ID_FIELD', 'ITEM_ID_FIELD', 'RATING_FIELD', 'TIME_FIELD',
    'seq_len',
    'LABEL_FIELD', 'threshold',
    'NEG_PREFIX',
    'ITEM_LIST_LENGTH_FIELD', 'LIST_SUFFIX', 'MAX_ITEM_LIST_LENGTH', 'POSITION_FIELD',
    'HEAD_ENTITY_ID_FIELD', 'TAIL_ENTITY_ID_FIELD', 'RELATION_ID_FIELD', 'ENTITY_ID_FIELD',
    'load_col', 'unload_col', 'unused_col', 'additional_feat_suffix',
    'rm_dup_inter', 'val_interval', 'filter_inter_by_user_or_item',
    'user_inter_num_interval', 'item_inter_num_interval',
    'alias_of_user_id', 'alias_of_item_id', 'alias_of_entity_id', 'alias_of_relation_id',
    'preload_weight', 'normalize_field', 'normalize_all',
    'benchmark_filename',
]


class Tips:
    def __init__(self):
        pass

    def get_dataloader(self):
        pass

    def create_basic_dataset(self, factory):
        dataset_module = importlib.import_module("tips.data.dataset")
        if hasattr(dataset_module, factory["model"] + "Dataset"):
            dataset_class = getattr(dataset_module, factory["model"] + "Dataset")
        else:
            model_type = factory["MODEL_TYPE"]
            type2class = {
                ModelType.GENERAL: "Dataset",
                ModelType.SEQUENTIAL: "SequentialDataset",
                ModelType.CONTEXT: "Dataset",
                ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
                ModelType.TRADITIONAL: "Dataset",
                ModelType.DECISIONTREE: "Dataset",
            }
            dataset_class = getattr(dataset_module, type2class[model_type])

        default_file = os.path.join(
            factory["checkpoint_dir"], f'{factory["dataset"]}-{dataset_class.__name__}.pth'
        )
        file = factory["dataset_save_path"] or default_file
        if os.path.exists(file):
            with open(file, "rb") as f:
                dataset = pickle.load(f)
            dataset_args_unchanged = True
            for arg in data_args + ["seed", "repeatable"]:
                if factory[arg] != dataset.factory[arg]:
                    dataset_args_unchanged = False
                    break
            if dataset_args_unchanged:
                logger = getLogger()
                logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
                return dataset

        dataset = dataset_class(factory)
        if factory["save_dataset"]:
            dataset.save()
        return dataset

    def create_dataset(self, factory):
        model_type = factory['MODEL_TYPE']
        dataset_module = importlib.import_module('tips_gnn.data.dataset')
        gen_graph_module_path = '.'.join(['tips_gnn.model.general_recommender', factory['model'].lower()])
        seq_module_path = '.'.join(['tips_gnn.model.sequential_recommender', factory['model'].lower()])
        if hasattr(dataset_module, factory['model'] + 'Dataset'):
            dataset_class = getattr(dataset_module, factory['model'] + 'Dataset')
        elif importlib.util.find_spec(gen_graph_module_path, __name__):
            dataset_class = getattr(dataset_module, 'GeneralGraphDataset')
        elif importlib.util.find_spec(seq_module_path, __name__):
            dataset_class = getattr(dataset_module, 'SessionGraphDataset')
        elif model_type == ModelType.SOCIAL:
            dataset_class = getattr(dataset_module, 'SocialDataset')
        else:
            return self.create_basic_dataset(factory)

        default_file = os.path.join(factory['checkpoint_dir'], f'{factory["dataset"]}-{dataset_class.__name__}.pth')
        file = factory['dataset_save_path'] or default_file
        if os.path.exists(file):
            with open(file, 'rb') as f:
                dataset = pickle.load(f)
            dataset_args_unchanged = True
            for arg in data_args + ['seed', 'repeatable']:
                if factory[arg] != dataset.factory[arg]:
                    dataset_args_unchanged = False
                    break
            if dataset_args_unchanged:
                logger = getLogger()
                logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
                return dataset

        dataset = dataset_class(factory)
        if factory['save_dataset']:
            dataset.save()
        return dataset

    def get_model_helper(self, model_name):
        model_submodule = [
            "general_recommender",
            "context_aware_recommender",
            "sequential_recommender",
            "knowledge_aware_recommender",
            "exlib_recommender",
        ]

        model_file_name = model_name.lower()
        model_module = None
        for submodule in model_submodule:
            module_path = ".".join(["tips.model", submodule, model_file_name])
            if importlib.util.find_spec(module_path, __name__):
                model_module = importlib.import_module(module_path, __name__)
                break

        if model_module is None:
            raise ValueError(
                "`model_name` [{}] is not the name of an existing model.".format(model_name)
            )
        model_class = getattr(model_module, model_name)
        return model_class

    def get_model(self, model_name):
        model_submodule = [
            'general_recommender', 'sequential_recommender', 'social_recommender'
        ]

        model_file_name = model_name.lower()
        model_module = None
        for submodule in model_submodule:
            module_path = '.'.join(['tips_gnn.model', submodule, model_file_name])
            if importlib.util.find_spec(module_path, __name__):
                model_module = importlib.import_module(module_path, __name__)
                break

        if model_module is None:
            model_class = self.get_model_helper(model_name)
        else:
            model_class = getattr(model_module, model_name)
        return model_class

    # def _get_customized_dataloader(factory, phase):
    #     if phase == 'train':
    #         return CustomizedTrainDataLoader
    #     else:
    #         eval_mode = factory["eval_args"]["mode"]
    #         if eval_mode == 'full':
    #             return CustomizedFullSortEvalDataLoader
    #         else:
    #             return CustomizedNegSampleEvalDataLoader

    def basic_data_preparation(self, factory, dataset):
        dataloaders = load_split_dataloaders(factory)
        if dataloaders is not None:
            train_data, valid_data, test_data = dataloaders
        else:
            model_type = factory["MODEL_TYPE"]
            built_datasets = dataset.build()

            train_dataset, valid_dataset, test_dataset = built_datasets
            train_sampler, valid_sampler, test_sampler = create_samplers(
                factory, dataset, built_datasets
            )

            # TODO: write dataloader

            if model_type != ModelType.KNOWLEDGE:
                train_data = self.get_dataloader(factory, "train")(
                    factory, train_dataset, train_sampler, shuffle=factory["shuffle"]
                )
            else:
                # TODO: write sampler
                kg_sampler = KGSampler(
                    dataset,
                    factory["train_neg_sample_args"]["distribution"],
                    factory["train_neg_sample_args"]["alpha"],
                )
                train_data = self.get_dataloader(factory, "train")(
                    factory, train_dataset, train_sampler, kg_sampler, shuffle=True
                )

            valid_data = self.get_dataloader(factory, "evaluation")(
                factory, valid_dataset, valid_sampler, shuffle=False
            )
            test_data = self.get_dataloader(factory, "evaluation")(
                factory, test_dataset, test_sampler, shuffle=False
            )
            if factory["save_dataloaders"]:
                save_split_dataloaders(
                    factory, dataloaders=(train_data, valid_data, test_data)
                )

        logger = getLogger()
        logger.info(
            set_color("[Training]: ", "pink")
            + set_color("train_batch_size", "cyan")
            + " = "
            + set_color(f'[{factory["train_batch_size"]}]', "yellow")
            + set_color(" train_neg_sample_args", "cyan")
            + ": "
            + set_color(f'[{factory["train_neg_sample_args"]}]', "yellow")
        )
        logger.info(
            set_color("[Evaluation]: ", "pink")
            + set_color("eval_batch_size", "cyan")
            + " = "
            + set_color(f'[{factory["eval_batch_size"]}]', "yellow")
            + set_color(" eval_args", "cyan")
            + ": "
            + set_color(f'[{factory["eval_args"]}]', "yellow")
        )
        return train_data, valid_data, test_data

    # def data_preparation(factory, dataset):
    #     seq_module_path = '.'.join(['tips_gnn.model.sequential_recommender', factory['model'].lower()])
    #     if importlib.util.find_spec(seq_module_path, __name__):
    #         # Special condition for sequential models of tips-Graph
    #         dataloaders = load_split_dataloaders(factory)
    #         if dataloaders is not None:
    #             train_data, valid_data, test_data = dataloaders
    #         else:
    #             built_datasets = dataset.build()
    #             train_dataset, valid_dataset, test_dataset = built_datasets
    #             train_sampler, valid_sampler, test_sampler = create_samplers(factory, dataset, built_datasets)
    #
    #             train_data = _get_customized_dataloader(factory, 'train')(factory, train_dataset, train_sampler, shuffle=True)
    #             valid_data = _get_customized_dataloader(factory, 'evaluation')(factory, valid_dataset, valid_sampler,
    #                                                                           shuffle=False)
    #             test_data = _get_customized_dataloader(factory, 'evaluation')(factory, test_dataset, test_sampler,
    #                                                                          shuffle=False)
    #             if factory['save_dataloaders']:
    #                 save_split_dataloaders(factory, dataloaders=(train_data, valid_data, test_data))
    #
    #         logger = getLogger()
    #         logger.info(
    #             set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
    #             set_color(f'[{factory["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
    #             set_color(f'[{factory["train_neg_sample_args"]}]', 'yellow')
    #         )
    #         logger.info(
    #             set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
    #             set_color(f'[{factory["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
    #             set_color(f'[{factory["eval_args"]}]', 'yellow')
    #         )
    #         return train_data, valid_data, test_data
    #     else:
    #         return basic_data_preparation(factory, dataset)

    def get_trainer_helper(self, model_type, model_name):
        try:
            return getattr(
                importlib.import_module("tips.trainer"), model_name + "Trainer"
            )
        except AttributeError:
            if model_type == ModelType.KNOWLEDGE:
                return getattr(importlib.import_module("tips.trainer"), "KGTrainer")
            elif model_type == ModelType.TRADITIONAL:
                return getattr(
                    importlib.import_module("tips.trainer"), "TraditionalTrainer"
                )
            else:
                return getattr(importlib.import_module("tips.trainer"), "Trainer")

    def get_trainer(self, model_type, model_name):
        try:
            return getattr(importlib.import_module('tips.trainer'), model_name + 'Trainer')
        except AttributeError:
            return self.get_trainer_helper(model_type, model_name)
