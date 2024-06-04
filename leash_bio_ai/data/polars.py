from abc import ABC, abstractmethod
import polars as pl
import logging


class PolarsPipeline(ABC):

    def __init__(self, logger_file):
        self.df = dataframe()
        self.logger_file = logger_file
        self.logger = setup_logger()

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
