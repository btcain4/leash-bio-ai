import polars as pl
import numpy as np

from leash_bio_ai.utils.conf import silver_logger_file
from leash_bio_ai.utils.conf import bronze_train_dir, bronze_test_dir
from leash_bio_ai.utils.conf import silver_train_dir, silver_test_dir
from leash_bio_ai.data.polars import PolarsPipeline, PipelineError


def df_sampler(df, proportion):
    """Returns a sample of a polars dataframes rows without replacement.

    Args:
        df (polars lazyframe): dataframe to return a sample of.
        proportion (float): number between 0 and 1 indicating the proportion of the dataset
                            to be sampled. (Function returns approxiamately this)
    """

    df_len = df.select(pl.len()).collect(streaming=True).item()

    np.random.seed(0)  # Replicable result
    probs = np.random.uniform(low=0, high=1, size=df_len)

    df = df.with_columns(pl.Series(name="probs", values=probs))
    samp_df = df.filter(pl.col("probs") <= proportion)
    samp_df = samp_df.drop("probs")
    samp_df = samp_df.collect(streaming=True)

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

    def __init__(self, logger_file, bronze_dir, silver_dir, test=False):
        super().__init__(logger_file)
        self.test = test
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.df = self.dataframe()
        self.df_slices = None

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

    def slice_df(self):

        df_len = self.df.select(pl.len()).collect(streaming=True).item()
        df_step = df_len / 10

        slices = np.arange(start=0, stop=df_len, step=df_step)

        for start in slices:
            (self.df_slices.append(self.df.slice(offset=start, length=df_step)))

    def downsample_slices(self):

        ds_dfs = []
        for i in range(len(self.df_slices)):

            pos_df = (
                self.df_slices[i].filter(pl.col("binds") == 1).collect(streaming=True)
            )
            neg_df = df_sampler(
                self.df_slices[i].filter(pl.col("binds") == 0), proportion=0.01
            )
            ds_dfs.append(pl.concat(items=[pos_df, neg_df], how="vertical"))

        return pl.concat(items=ds_dfs, how="vertical")

    def execute(self):

        try:
            if self.test:

                self.logger.info("Imputing Protein Names (Test Set)")
                self.protein_imputation()

                self.logger.info("Performing Column Corrections (Test Set)")
                self.column_correction()

                self.logger.info("Save to Parquet (Test Set)")
                self.df.write_parquet(file=self.silver_dir)

            else:

                self.logger.info("Imputing Protein Names (Train Set)")
                self.protein_imputation()

                self.logger.info("Performing Column Corrections (Train Set)")
                self.column_correction()

                self.logger.info("Slice Dataframe (Train Set)")
                self.slice_df()

                self.logger.info("Performing Downsampling (Train Set)")
                downsampled_df = self.downsample_slices()

                self.logger.info("Save to Parquet (Train Set)")
                downsampled_df.write_parquet(file=self.silver_dir)

        except:
            self.logger.info("An error occured in the pipeline")
            raise PipelineError("An error occured in the pipeline")
