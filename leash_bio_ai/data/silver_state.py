import polars as pl

from leash_bio_ai.utils.conf import silver_logger_file
from leash_bio_ai.utils.conf import data_dir, bronze_train_dir, bronze_test_dir
from leash_bio_ai.data.polars import PolarsPipeline, PipelineError


def df_sampler(df, proportion):
    """Returns a sample of a polars dataframes rows without replacement.

    Args:
        df (polars dataframe): dataframe to return a sample of.
        proportion (float): number between 0 and 1 indicating the proportion of the dataset
                            to be sampled.
    """
    pass


class SilverPipeline(PolarsPipeline):
    """PolarsPipeline child class that setups and executes the data pipeline
       to transform the data from the bronze to the silver layers

    Attributes:
        bronze_dir (str): string specifying the location of the bronze layer data
        df (polars lazyframe): bronze layer dataframe to be transformed by the pipeline
        Check polars.py for attributes inherited from the PolarsPipeline abstract class
    """

    def __init__(self, logger_file, bronze_dir):
        super().__init__(logger_file)
        self.bronze_dir = bronze_dir
        self.df = self.dataframe()

    def dataframe(self):
        df = pl.scan_parquet(source=self.bronze_dir)
        return df

    def protein_imputation(self):
        """Integer imputation on protein_name column to reduce dataset memory stress on local compute"""
        mapper = {"BRD4": 0, "HSA": 1, "sEH": 2}
        self.df = self.df.with_columns(
            protein_int=pl.col("protein_name").replace(mapper)
        )

    def column_correction(self):
        """Column reduction and integer type correction to reduce dataset size and dataset memory stress on local compute"""
        self.df = self.df.select(
            "id",
            "molecule_smiles",
            pl.col("protein_int").cast(pl.Int8),
            pl.col("binds").cast(pl.Int8),
        )

    def execute(self):
        pass
