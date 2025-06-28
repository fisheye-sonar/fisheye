import os

from analysis.compare_2_txt_files import compare_2_txt_files
from analysis.crop_generated_txt_to_frame_range import crop_txt_to_frame_range
from analysis.mot_to_txt import mot_to_txt


dataset_dir = "/home/mahobley/Data/CFC22/restructured_dataset"
model_output_dir = "/home/mahobley/Code/fisheye/results"

generation_dir = "/home/mahobley/Code/fisheye/analysis/generated_results"
gt_dir = "/home/mahobley/Code/fisheye/analysis/gt_files"


annotations_dir = os.path.join(dataset_dir, "annotations")
info_dir = os.path.join(dataset_dir, "info")

annotations_file_name = f"gt.txt"

analysis_output_dir = "/home/mahobley/Code/fisheye/analysis/outputs"

locations = ["nushagak"]

for location in locations:

    os.makedirs(os.path.join(gt_dir, location), exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, location), exist_ok=True)
    os.makedirs(os.path.join(generation_dir, location), exist_ok=True)
    os.makedirs(os.path.join(analysis_output_dir, location), exist_ok=True)

    clips = os.listdir(os.path.join(annotations_dir, location))
    # clips = ["RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_900_1200"]

    for clip_name in clips:
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

        input_path = os.path.join(model_output_dir, input_fn)
        output_path = os.path.join(generation_dir, location, output_fn)
        output_filepath = os.path.join(analysis_output_dir, location, clip_name)

        # convert the MOT annotations to a txt file with the same format as the Fisheye generated txt file
        x = mot_to_txt(annotations_path, info_path, start_frame, output_gt_path)

        if os.path.exists(input_path):
            # crop the Fisheye generated txt file to the frame range
            crop_txt_to_frame_range(input_path, start_frame, end_frame, output_path)

            # compare the Fisheye generated txt file to the MOT annotations

            compare_2_txt_files(
                gt_file=output_gt_path,
                pred_file=output_path,
                output_filepath=output_filepath,
                plot=True,
                save=True,
                max_frame_diff=10,
                max_r_diff=0.2,
                remove_multiple_tracks=False,
            )
        else:
            print(f"No input file found for {clip_name}")
