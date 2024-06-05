from abc import ABC, abstractmethod
import polars as pl
import logging


class PolarsPipeline(ABC):
    """Abstract class to provide structure for polar pipeline classes

    Attributes:
        df (polars dataframe or lazyframe): result of dataframe() abstract method
        logger_file (str): string specifying location of desired logging file
        logger (obj): logger resultant from setup_logger() method
    """

    def __init__(self, logger_file):
        self.df = self.dataframe()
        self.logger_file = logger_file
        self.logger = self.setup_logger()

    @abstractmethod
    def dataframe(self):
        pass

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=self.logger_file, level=logging.DEBUG)
        logger.info("--- Logging Started ---")
        return logger

    @abstractmethod
    def execute(self):
        pass
