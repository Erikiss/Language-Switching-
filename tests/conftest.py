import os
import pytest
import torch

from lsne.config import ModelSpec
from lsne.model import load_model

TEST_MODEL = os.environ.get("LSNE_TEST_MODEL", "EleutherAI/pythia-14m")
TEST_REVISION = os.environ.get("LSNE_TEST_REVISION", "step98000")


@pytest.fixture(scope="session")
def spec():
    return ModelSpec(name=TEST_MODEL, revision=TEST_REVISION, dtype="float32", device="cpu")


@pytest.fixture(scope="session")
def mb(spec):
    torch.manual_seed(0)
    return load_model(spec)


@pytest.fixture(scope="session")
def tok(mb):
    return mb.tokenizer


@pytest.fixture(scope="session")
def sample_ids(mb):
    return mb.encode(" wanted to mediate this in reports and parables, in relation to which the reflections are simply")
