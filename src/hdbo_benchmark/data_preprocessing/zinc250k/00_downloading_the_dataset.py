"""Downloads Zinc250k using torch drug

Downloads the Zinc250k dataset using torch drug, and
saves it on the assets folder.
"""

import pickle
from pathlib import Path

try:
    from torchdrug import datasets  # type: ignore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "TorchDrug is required to download Zinc250k for preprocessing. "
        "This repository's preprocessing environment pins Python 3.8 and "
        "torchdrug==0.2.1 in environment.data_preprocessing.yml. "
        "If you are running in Colab on Python 3.12, prefer creating the "
        "processed Zinc files outside Colab and then copying them into Drive."
    ) from exc

if __name__ == "__main__":
    # The root of the project.
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()

    DATASET_DIR = ROOT_DIR / "data" / "small_molecule_datasets" / "raw"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Download the dataset.
    zinc250k = datasets.ZINC250k(DATASET_DIR, kekulize=True, atom_feature="symbol")

    # Save the dataset.
    with open(DATASET_DIR / "zinc250k.pkl", "wb") as fout:
        pickle.dump(zinc250k, fout)
