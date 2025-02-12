from conftest import ARIS_FILE

from fisheye.config import ARISDatasetConfig
from utils.visualisation_utils import generate_echogram_gif_from_aris


def test_generate_echogram_gif_from_small_aris():
    config = ARISDatasetConfig(
        aris_filepath=ARIS_FILE,
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
        aris_filepath=ARIS_FILE,
        return_unwarped=False,
        return_echogram=True,
    )

    generate_echogram_gif_from_aris(
        config,
        save_filename="",
        echogram_pop=False,
        return_unwarped=False,
    )
