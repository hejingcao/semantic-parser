# -*- coding: utf-8 -*-

from framework.torch_extra.train_session import TrainSession
from model_utils import get_model_class


def train(model_class):
    session = TrainSession.from_command_line(model_class)
    session.run()


if __name__ == '__main__':
    train(model_class=get_model_class())
