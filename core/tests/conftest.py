import numpy as np
import pytest

from bayesian_credit.config import ArmSpaceConfig, SegmentationConfig
from bayesian_credit.data.arms import ArmSpace
from bayesian_credit.data.segmentation import Segmenter


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def arm_space() -> ArmSpace:
    return ArmSpace.from_config(ArmSpaceConfig())


@pytest.fixture
def small_arm_space() -> ArmSpace:
    return ArmSpace.from_grid([5_000, 15_000], [0.10, 0.20], [36], include_null=True)


@pytest.fixture
def segmenter() -> Segmenter:
    return Segmenter.from_config(SegmentationConfig())
