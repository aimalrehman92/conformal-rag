"""
Factory functions for selecting a conformal implementation.

Supported implementations:

- ``author``: faithful preservation of the public repository behavior.
- ``aimal``: corrected conformal implementation.

The factory only selects calibration behavior. It does not affect retrieval,
generation, scoring, verification, or upstream cache identity.
"""

from src.calibration.author_conditional_conformal import (
    AuthorGroupConditionalConformal,
)
from src.calibration.author_conformal import (
    AuthorSplitConformalCalibration,
)
from src.calibration.conditional_conformal import (
    GroupConditionalConformal,
)
from src.calibration.conformal import SplitConformalCalibration

SUPPORTED_CONFORMAL_IMPLEMENTATIONS = {
    "author",
    "aimal",
}


def _validate_implementation(implementation: str) -> None:
    if implementation not in SUPPORTED_CONFORMAL_IMPLEMENTATIONS:
        raise ValueError(
            "Unknown conformal implementation "
            f"{implementation!r}. Supported implementations are: "
            f"{sorted(SUPPORTED_CONFORMAL_IMPLEMENTATIONS)}"
        )


def create_split_conformal_calibration(
    implementation: str,
    dataset_name: str,
    runs: int,
    seed: int,
):
    """
    Create the requested split-conformal implementation.

    ``seed`` is intentionally ignored by the author implementation because
    the public repository uses Python's global random state instead of a
    calibration-local deterministic RNG.
    """
    _validate_implementation(implementation)

    if implementation == "author":
        return AuthorSplitConformalCalibration(
            dataset_name=dataset_name,
            runs=runs,
        )

    return SplitConformalCalibration(
        dataset_name=dataset_name,
        runs=runs,
        seed=seed,
    )


def create_group_conditional_calibration(
    implementation: str,
    dataset_name: str,
    result_dir: str,
    runs: int,
    seed: int,
):
    """
    Create the requested group-conditional conformal implementation.

    ``seed`` is intentionally ignored by the author implementation.
    """
    _validate_implementation(implementation)

    if implementation == "author":
        return AuthorGroupConditionalConformal(
            dataset_name=dataset_name,
            result_dir=result_dir,
            runs=runs,
        )

    return GroupConditionalConformal(
        dataset_name=dataset_name,
        result_dir=result_dir,
        runs=runs,
        seed=seed,
    )
