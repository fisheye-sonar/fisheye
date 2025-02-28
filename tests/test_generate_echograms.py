from conftest import ARIS_FILE

from fisheye.configs import ARISDatasetConfig
from utils.visualisation_utils import generate_echogram_gif_from_aris


def test_generate_echogram_gif_from_small_aris():
    config = ARISDatasetConfig(
        filepath=ARIS_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=True,
        return_unwarped=False,
    )


def test_generate_echogram_gif_from_small_aris_filter_kernel():
    config = ARISDatasetConfig(
        filepath=ARIS_FILE,
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
    config = ARISDatasetConfig(
        filepath=ARIS_FILE,
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
    config = ARISDatasetConfig(
        filepath=ARIS_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=False,
        return_unwarped=False,
        echogram_filter_kernel=7,
        echogram_filter_tol=0.15,
    )
