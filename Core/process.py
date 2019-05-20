# -*- coding: UTF-8 -*-
import abc

class ProcessBase:

    @abc.abstractmethod
    def process(self):
        pass

    @abc.abstractmethod
    def _train_process(self):
        pass

    @abc.abstractmethod
    def _val_process(self):
        pass

    @abc.abstractmethod
    def _test_process(self):
        pass