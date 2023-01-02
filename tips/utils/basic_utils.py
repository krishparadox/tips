from enum import Enum
import os


class ModelType(Enum):
    GENERAL = 1
    SEQUENTIAL = 2
    CONTEXT = 3
    KNOWLEDGE = 4
    TRADITIONAL = 5
    DECISIONTREE = 6


class InputType(Enum):
    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
