from abc import ABC, abstractmethod
import polars as pl


class PolarsPipeline(ABC):

    def __init__(self):
        self.ldf = lazyframe()
        self.logger = 0

    @abstractmethod
    def lazyframe(self):
        pass

    @abstractmethod
    def execute(self):
        pass
