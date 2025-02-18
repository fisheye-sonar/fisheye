import pytest

from conftest import DDF_FILE

from fisheye.dataclasses import ARISDatasetConfig
from utils.visualisation_utils import generate_echogram_gif_from_aris

"""
Same as test_generate_echograms but using didson version 3 file type"""


def test_generate_echogram_gif_from_small_aris():
    config = ARISDatasetConfig(
        filepath=DDF_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=True,
        return_unwarped=False,
    )


def test_generate_echogram_gif_from_small_aris_no_echo_pop():
    config = ARISDatasetConfig(
        filepath=DDF_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=False,
        return_unwarped=False,
    )
