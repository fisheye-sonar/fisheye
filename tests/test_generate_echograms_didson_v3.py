from conftest import DDF_FILE

from fisheye.configs import BaseDatasetConfig
from utils.visualisation_utils import generate_echogram_gif_from_aris

"""Same as test_generate_echograms but using didson version 3 file type"""


def test_generate_echogram_gif_from_small_aris():
    config = BaseDatasetConfig(
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


def test_generate_echogram_gif_from_small_aris_filter():
    config = BaseDatasetConfig(
        filepath=DDF_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=True,
        return_unwarped=False,
        echogram_filter_kernel=7,
        echogram_filter_tol=0.15,
    )


def test_generate_echogram_gif_from_small_aris_no_echo_pop():
    config = BaseDatasetConfig(
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


def test_generate_echogram_gif_from_small_aris_no_echo_pop_filtered():
    config = BaseDatasetConfig(
        filepath=DDF_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=False,
        return_unwarped=False,
        resize_mode="scale",
        echogram_filter_kernel=7,
        echogram_filter_tol=0.15,
    )


def test_generate_echogram_gif_from_small_aris_no_echo_pop_filtered_pad():
    config = BaseDatasetConfig(
        filepath=DDF_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=False,
        return_unwarped=False,
        resize_mode="pad",
        echogram_filter_kernel=7,
        echogram_filter_tol=0.15,
    )


test_generate_echogram_gif_from_small_aris_no_echo_pop_filtered()
