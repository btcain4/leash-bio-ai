import polars as pl
import numpy as np

from leash_bio_ai.utils.conf import silver_logger_file
from leash_bio_ai.utils.conf import bronze_train_dir, bronze_test_dir
from leash_bio_ai.utils.conf import silver_train_dir, silver_test_dir
from leash_bio_ai.data.pipeline import PolarsPipeline, PipelineError


def df_sampler(df, proportion):
    """Returns a sample of a polars dataframes rows without replacement.

    Args:
        df (polars dataframe): dataframe to return a sample of.
        proportion (float): number between 0 and 1 indicating the proportion of the dataset
                            to be sampled. (Function returns approxiamately this)
    """

    df_len = df.select(pl.len()).item()

    np.random.seed(0)  # Replicable result
    probs = np.random.uniform(low=0, high=1, size=df_len)

    df = df.with_columns(pl.Series(name="probs", values=probs))
    samp_df = df.filter(pl.col("probs") <= proportion)
    samp_df = samp_df.drop("probs")
    samp_df = samp_df

    return samp_df


class SilverPipeline(PolarsPipeline):
    """PolarsPipeline child class that setups and executes the data pipeline
       to transform the data from the bronze to the silver layers

    Attributes:
        bronze_dir (str): string specifying the location of the bronze layer data
        df (polars lazyframe): bronze layer dataframe to be transformed by the pipeline
        test_set (boolean): indicates if data being processed by the pipeline is the test set
        Check polars.py for attributes inherited from the PolarsPipeline abstract class
    """

    def __init__(self, logger_file, bronze_dir, silver_dir, slice=None, test=False):
        super().__init__(logger_file)
        self.test = test
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.slice = slice
        self.df = self.dataframe()

    def dataframe(self):
        if self.test:
            df = pl.scan_parquet(source=self.bronze_dir)
        else:
            df = pl.scan_parquet(source=self.bronze_dir, low_memory=True)
            df = df.filter(pl.col("id") >= self.slice[0])
            df = df.filter(pl.col("id") <= self.slice[1])
            df = df.collect(streaming=True)

        return df

    def protein_imputation(self):
        """Integer imputation on protein_name column to reduce dataset memory stress on local compute"""

        mapper = {"BRD4": 0, "HSA": 1, "sEH": 2}
        self.df = self.df.with_columns(
            protein_int=pl.col("protein_name").replace(mapper)
        )

    def column_correction(self):
        """Column reduction and integer type correction to reduce dataset size and dataset memory stress on local compute"""

        if self.test:
            self.df = self.df.select(
                "id", "molecule_smiles", pl.col("protein_int").cast(pl.Int8)
            )

        else:
            self.df = self.df.select(
                "id",
                "molecule_smiles",
                pl.col("protein_int").cast(pl.Int8),
                pl.col("binds").cast(pl.Int8),
            )

    def downsample(self):

        pos_df = self.df.filter(pl.col("binds") == 1)
        neg_df = df_sampler(self.df.filter(pl.col("binds") == 0), proportion=0.01)
        self.df = pl.concat(items=[pos_df, neg_df], how="vertical")

        del pos_df
        del neg_df

    def save_downsample(self):

        if self.slice[0] == 0:
            self.df.write_parquet(file=self.silver_dir)
        else:
            silver_df = pl.read_parquet(source=self.silver_dir)
            self.df = pl.concat(items=[silver_df, self.df], how="vertical")
            self.df.write_parquet(file=self.silver_dir)

    def execute(self):

        try:
            if self.test:

                self.logger.info("Imputing Protein Names (Test Set)")
                self.protein_imputation()

                self.logger.info("Performing Column Corrections (Test Set)")
                self.column_correction()

                self.logger.info("Save to Parquet (Test Set)")
                self.df.collect().write_parquet(file=self.silver_dir)

                self.logger.info("Silver Test Set Generated")

            else:

                self.logger.info("Imputing Protein Names (Train Set)")
                self.protein_imputation()

                self.logger.info("Performing Column Corrections (Train Set)")
                self.column_correction()

                self.logger.info("Downsampling Negative Binds (Train Set)")
                self.downsample()

                self.logger.info("Save to Parquet (Train Set)")
                self.save_downsample()

                self.logger.info(f"{self.slice} Silver Train Set Generated")

        except:
            self.logger.info("An error occured in the pipeline")
            raise PipelineError("An error occured in the pipeline")


if __name__ == "__main__":

    otestSilverPipeline = SilverPipeline(
        logger_file=silver_logger_file,
        bronze_dir=bronze_test_dir,
        silver_dir=silver_test_dir,
        test=True,
    )
    otestSilverPipeline.execute()
    del otestSilverPipeline

    ## Batch process training set into 30 different sets
    slices = np.arange(start=0, stop=295246830, step=9841561)
    for slice in slices:

        otrainSilverPipeline = SilverPipeline(
            logger_file=silver_logger_file,
            bronze_dir=bronze_train_dir,
            silver_dir=silver_train_dir,
            slice=[slice, slice + 9841561],
            test=False,
        )
        otrainSilverPipeline.execute()
        del otrainSilverPipeline
