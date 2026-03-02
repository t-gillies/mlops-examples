from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset and perform basic validation.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the dataset is missing a ``target`` column.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Did you run `dvc pull` (or generate + dvc add)?"
        )

    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(f"Dataset at {path} is missing a 'target' column.")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into stratified train/test sets.

    Parameters
    ----------
    df : DataFrame
        Must contain a ``target`` column.
    test_size : float
        Fraction of data reserved for the test set.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
