import os
import json

from analysis.compare_2_txt_files import compare_2_txt_files
from analysis.crop_generated_txt_to_frame_range import crop_txt_to_frame_range
from analysis.convert_mot_to_txt import mot_to_txt
from analysis.analyse_prediction_errors import analyse_location, read_jsons_for_location
from analysis.plot_errors import plot_metrics_across_locations
from tqdm import tqdm

dataset_dir = "/home/mahobley/Data/CFC22/restructured_dataset"
model_output_dir = "/home/mahobley/Code/fisheye/results"

generation_dir = "/home/mahobley/Code/fisheye/analysis/generated_results"
gt_dir = "/home/mahobley/Code/fisheye/analysis/gt_files"
summary_json_dir = "/home/mahobley/Code/fisheye/analysis/summary_stats"


annotations_dir = os.path.join(dataset_dir, "annotations")
info_dir = os.path.join(dataset_dir, "info")

annotations_file_name = f"gt.txt"

analysis_output_dir = "/home/mahobley/Code/fisheye/analysis/outputs"

locations_dirs = {
    "nushagak": "left",
    "elwha": "right",
    "kenai-rightbank": "left",
    "kenai-val": "right",
    "kenai-train": "right",
    "kenai-channel": "right",
}

# this is so you dont have to run all the steps every time (saves time for analysis)
steps = [
    # "convert_mot_to_txt",
    # "crop_txt_to_frame_range",
    # "compare_2_txt_files",
    # "analyse_location",
    "plot_metrics_across_locations",
]

metrics_dict = {}
for location, upstream_direction in locations_dirs.items():
    location_info_dicts = []

    os.makedirs(os.path.join(gt_dir, location), exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, location), exist_ok=True)
    os.makedirs(os.path.join(generation_dir, location), exist_ok=True)
    os.makedirs(os.path.join(analysis_output_dir, location), exist_ok=True)

    clips = os.listdir(os.path.join(annotations_dir, location))

    for clip_name in tqdm(clips, leave=False):
        aris_name = "_".join(clip_name.split("_")[:-2])
        info_name = aris_name

        annotations_path = os.path.join(
            annotations_dir, location, clip_name, annotations_file_name
        )
        info_path = os.path.join(info_dir, location, info_name, f"{info_name}.json")

        start_frame = int(clip_name.split("_")[-2])
        end_frame = int(clip_name.split("_")[-1])

        output_fn = f"{clip_name}_gt.txt"
        input_fn = f"FCe_{aris_name}_ID_.txt"
        output_fn = f"{input_fn.split('.')[0]}_{start_frame}_{end_frame}_cropped.txt"
        output_gt_path = os.path.join(gt_dir, location, output_fn)

        input_path = os.path.join(model_output_dir, location, input_fn)
        output_path = os.path.join(generation_dir, location, output_fn)
        output_filepath = os.path.join(analysis_output_dir, location, clip_name)

        if "convert_mot_to_txt" in steps:
            # convert the MOT annotations to a txt file with the same format as the Fisheye generated txt file
            x = mot_to_txt(
                annotations_path,
                info_path,
                start_frame,
                upstream_direction,
                output_gt_path,
            )

        if os.path.exists(input_path):
            if "crop_txt_to_frame_range" in steps:
                # crop the Fisheye generated txt file to the frame range
                crop_txt_to_frame_range(input_path, start_frame, end_frame, output_path)
            if not os.path.exists(output_gt_path):
                print(
                    f"\n\033[33mNo gt_file file found for {clip_name} {output_gt_path}\033[0m"
                )
                continue
            if not os.path.exists(output_path):
                print(
                    f"\n\033[33mNo pred_file file found for {clip_name} {output_path}\033[0m"
                )
                continue

            if "compare_2_txt_files" in steps:
                # compare the Fisheye generated txt file to the MOT annotations
                info_dict = compare_2_txt_files(
                    file_A=output_gt_path,
                    file_B=output_path,
                    output_filepath=output_filepath,
                    plot=True,
                    save_plots_per_clip=True,
                    save_json_per_clip=True,
                    max_frame_diff=10,
                    max_r_diff=0.2,
                    remove_multiple_crossings_per_track=False,
                )
                location_info_dicts.append(info_dict)
        else:
            print(f"No input file found for {clip_name} {input_path}")

    #  analyse a locations predictions vs the ground truth, this also generates the plots for each location
    if "analyse_location" in steps:
        if "compare_2_txt_files" not in steps:
            location_info_dicts = read_jsons_for_location(analysis_output_dir, location)
        metrics = analyse_location(
            location_info_dicts,
            location,
            print_for_latex=False,
            plot=True,
            save_json_path=summary_json_dir,
        )
        metrics_dict[location] = metrics

# plot a graph of the metrics per location against each other
if "plot_metrics_across_locations" in steps:
    if "analyse_location" not in steps:
        metrics_dict = {}

        for location in locations_dirs.keys():
            loc_metric_path = os.path.join(summary_json_dir, f"{location}.json")
            with open(loc_metric_path, "r") as f:
                metrics_dict[location] = json.load(f)

    plot_metrics_across_locations(
        metrics_dict,
        "analysis/figures/error_percent_by_location.png",
        metrics=["nMAE", "nMNE", "nMANE"],
    )
