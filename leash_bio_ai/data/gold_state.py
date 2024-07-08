import polars as pl

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from leash_bio_ai.utils.conf import gold_logger_file
from leash_bio_ai.utils.conf import silver_train_dir, silver_test_dir
from leash_bio_ai.utils.conf import gold_train_dir, gold_test_dir

from leash_bio_ai.features.utils import gold_features

from leash_bio_ai.data.pipeline import PolarsPipeline, PipelineError


def gen_bit_vec(fpgen, molecule_smile):
    mol = Chem.MolFromSmiles(molecule_smile)
    bit_vec = fpgen.GetFingerprint(mol)
    return list(bit_vec)


def gen_mol_desc(molecule_smile):
    mol = Chem.MolFromSmiles(molecule_smile)
    return [
        Descriptors.ExactMolWt(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.MolWt(mol),
        Descriptors.NumValenceElectrons(mol),
    ]


class GoldPipeline(PolarsPipeline):
    """_summary_

    Attributes:
        attr (_type_): _description_
    """

    def __init__(self, logger_file, silver_dir, gold_dir, gold_features, test=False):
        super().__init__(logger_file)
        self.test = test
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir
        self.gold_features = gold_features
        self.df = self.dataframe()

    def dataframe(self):
        df = pl.scan_parquet(source=self.silver_dir)
        return df

    def fingerprint_generator(self):
        self.fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=500)

    def molecule_smiles(self):
        self.smiles = self.df["molecule_smiles"].to_numpy()

    def generate_bit(self):
        self.bit_vecs = [gen_bit_vec(self.fpgen, molecule_smile=x) for x in self.smiles]

    def generate_bit_feats(self):
        bit_df = pl.DataFrame(
            data=[pl.Series("bit_vecs", self.bit_vecs)],
            schema={"bit_vecs": pl.Array(pl.Int8, 500)},
        )
        self.df = pl.concat(items=[self.df, bit_df], how="horizontal")

        self.df = self.df.with_columns(
            (pl.col("bit_vecs").arr.sum()).alias("bit_vecs_sum")
        )
        self.df = self.df.with_columns(
            (pl.col("bit_vecs").arr.std()).alias("bit_vecs_std")
        )
        self.df = self.df.with_columns(
            (pl.col("bit_vecs").arr.var()).alias("bit_vecs_var")
        )

        bit_pos_cols = [f"pos_{x}" for x in range(100)]
        bit_pos_df = pl.DataFrame(
            data=self.df.select("bit_vecs").to_numpy(), schema=bit_pos_cols
        )
        self.df = pl.concat(items=[self.df, bit_pos_df], how="horizontal")

    def generate_mol_descs(self):
        self.mol_descs = [gen_mol_desc(x) for x in self.smiles]

    def generate_mol_descs_feats(self):
        descs_cols = [
            "ExactMolWt",
            "FpDensityMorgan1",
            "FpDensityMorgan2",
            "FpDensityMorgan3",
            "HeavyAtomMolWt",
            "MolWt",
            "NumValenceElectrons",
        ]
        mol_descs_df = pl.DataFrame(data=self.mol_descs, schema=descs_cols)
        self.df = pl.concat(items=[self.df, mol_descs_df])

    def select_features(self):
        if test:
            self.df = self.df.select(gold_features + ["id"])
        else:
            self.df = self.df.select(gold_features + ["binds", "id"])

    def execute(self):

        try:
            self.logger.info("Creating Fingerprint Generator")
            self.fingerprint_generator()

            self.logger.info("Collecting Molecule Smiles Array")
            self.molecule_smiles()

            self.logger.info("Generating Molecule Bit Vectors")
            self.generate_bit()

            self.logger.info("Generating Molecule Bit Vector Features")
            self.generate_bit_feats()

            self.logger.info("Generating Molecule Descriptions")
            self.generate_mol_descs()

            self.logger.info("Generating Molecule Descriptions Features")
            self.generate_mol_descs_feats()

            self.logger.info("Trimming to Selected Model Features")
            self.select_features()

            self.logger.info("Saving Gold Layer Dataframe")
            self.df.write_parquet(file=self.gold_dir)

        except:
            self.logger.info("An error occured in the pipeline")
            raise PipelineError("An error occured in the pipeline")


if __name__ == "__main__":

    otestGoldPipeline = GoldPipeline(
        logger_file=gold_logger_file,
        silver_dir=silver_test_dir,
        gold_dir=gold_test_dir,
        gold_features=gold_features,
        test=True,
    )
    otestGoldPipeline.execute()
    del otestGoldPipeline

    otrainGoldPipeline = GoldPipeline(
        logger_file=gold_logger_file,
        silver_dir=silver_train_dir,
        gold_dir=gold_train_dir,
        gold_features=gold_features,
        test=False,
    )
    otrainGoldPipeline.execute()
    del otrainGoldPipeline
