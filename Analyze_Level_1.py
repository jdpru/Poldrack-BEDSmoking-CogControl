#!/usr/bin/env python3

import glob
import numpy as np
import pandas as pd
import json
import sys
import os
from nilearn.image import smooth_img
from argparse import ArgumentParser, RawTextHelpFormatter
from utils_lev1.qa1 import (
    update_excluded_subject_csv,
    qa_design_matrix,
    add_to_html_summary,
)


def get_confounds_aroma_nonaggr_data(confounds_file):
    """
    Creates nuisance regressors for the nonaggressive denoised AROMA output
      from fmriprep.
    input:
      confounds_file: path to confounds file from fmriprep
    output:
      confound_regressors: includes WM, CSF, dummies to model nonsteady volumes
                           as they are not smoothed, cosine basis set (req with
                           AROMA use)
      percent_high_motion:  Percentage of high motion time points.  High motion
                            is defined by the following
                            FD>.5, stdDVARS>1.2 (that relates to DVARS>.5)
    """
    confounds_df = pd.read_csv(confounds_file, sep="\t", na_values=["n/a"]).fillna(0)
    excessive_movement = (confounds_df.framewise_displacement > 0.5) | (
        confounds_df.std_dvars > 1.5
    )
    percent_high_motion = np.mean(excessive_movement)

    # confounds = confounds_df.filter(regex='non_steady|cosine|^csf$|^white_matter$').copy()

    confounds = confounds_df.filter(
        regex="non_steady|cosine|framewise_displacement"
        "|trans_x$|trans_x_derivative1$|trans_x_power2$|trans_x_derivative1_power2$"
        "|trans_y$|trans_y_derivative1$|trans_y_power2$|trans_y_derivative1_power2$"
        "|trans_z$|trans_z_derivative1$|trans_z_power2$|trans_z_derivative1_power2$"
        "|rot_x$|rot_x_derivative1$|rot_x_power2$|rot_x_derivative1_power2$"
        "|rot_y$|rot_y_derivative1$|rot_y_power2$|rot_y_derivative1_power2$"
        "|rot_z$|rot_z_derivative1$|rot_z_power2$|rot_z_derivative1_power2$"
    ).copy()
    return confounds, percent_high_motion


def get_nscans(timeseries_data_file):
    """
    Get the number of time points from 4D data file
    input: time_series_data_file: Path to 4D file
    output: nscans: number of time points
    """
    import nibabel as nb

    fmri_data = nb.load(timeseries_data_file)
    n_scans = fmri_data.shape[3]
    return n_scans

# def get_tr(root, task):
#     """
#     Get the TR from the bold json file
#     input:
#         root: Root for BIDS data directory
#         task: Task name
#     output: TR as reported in json file (presumable in s)
#     """
#     json_file = glob.glob(f"{root}/*{task}_bold.json")[0]
#     print(json_file)
#     print(task)
#     with open(json_file, "rb") as f:
#         task_info = json.load(f)
#     tr = task_info["RepetitionTime"]
#     return tr

# def get_tr(root, task):
#     """
#     Get the TR from the bold json file
#     input:
#         root: Root for BIDS data directory
#         task: Task name
#     output: TR as reported in json file (presumable in s)
#     """

#     json_file = glob.glob(f"{root}/*{task}_bold.json")[0]
#     with open(json_file, "rb") as f:
#         task_info = json.load(f)
#     tr = task_info["RepetitionTime"]
#     return tr
    
# "old school version" testing out
def get_tr(root, task):
    """
    Get the TR from the bold json file
    input:
        root: Root for BIDS data directory
        task: Task name (like 'stopSignal' or 'discountFix')
    output: TR as reported in json file (presumably in seconds)
    """
    # Construct the expected filename based on the task
    expected_filename = "task-{}_bold.json".format(task)
    json_file = os.path.join(root, expected_filename)

    # Check if the file exists
    if not os.path.isfile(json_file):
        raise FileNotFoundError("No JSON file found for task '{}' in '{}'. Please check if the task name is correct.".format(task, root))

    with open(json_file, "r") as f:  # Use "r" to read JSON
        task_info = json.load(f)

    tr = task_info["RepetitionTime"]  # Assuming RepetitionTime exists in the JSON
    return tr


def make_desmat_contrasts(
    root, task, events_file, add_deriv, n_scans, confounds_file=None, regress_rt="no_rt"
):
    """
    Creates design matrices and contrasts for each task.  Should work for any
    style of design matrix as well as the regressors are defined within
    the imported make_task_desmat_fcn_map (dictionary of functions).
    A single RT regressor can be added using regress_rt='rt_uncentered'
    Input:
        root:  Root directory (for BIDS data)
        task: Task name
        events_file: File path to events.tsv for the given task
        add_deriv: 'deriv_yes' or 'deriv_no', recommended to use 'deriv_yes'
        n_scans: Number of scans
        confound_file (optional): File path to fmriprep confounds file
        regress_rt: 'no_rt' or 'rt_uncentered' or 'rt_centered'
    Output:
        design_matrix, contrasts: Full design matrix and contrasts for nilearn model
        percent junk: percentage of trials labeled as "junk".  Used in later QA.
        percent high motion: percentage of time points that are high motion.  Used later in QA.
    """
    from utils_lev1.first_level_designs import make_task_desmat_fcn_dict

    if confounds_file is not None:
        confound_regressors, percent_high_motion = get_confounds_aroma_nonaggr_data(
            confounds_file
        )
    else:
        confound_regressors = None

    tr = get_tr(root, task)

    design_matrix, contrasts, percent_junk, events_df = make_task_desmat_fcn_dict[task](
        events_file, add_deriv, regress_rt, n_scans, tr, confound_regressors
    )
    return design_matrix, contrasts, percent_junk, percent_high_motion, tr, events_df


def check_file(glob_out):
    """
    Checks if file exists
    input:
        glob_out: output from glob call attempting to retreive files.  Note this
        might be simplified for other data.  Since the tasks differed between sessions
        across subjects, the ses directory couldn't be hard coded, in which case glob
        would not be necessary.
    output:
        file: Path to file, if it exists
        file_missing: Indicator for whether or not file exists (used in later QA)
    """
    if len(glob_out) > 0:
        file = glob_out[0]
        file_missing = [0]
    else:
        file = []
        file_missing = [1]
    return file, file_missing

# older version that does not include sessions
# def get_files(root, subid, task):
#     """Fetches files (events.tsv, confounds, mask, data)
#     if files are not present, excluded_subjects.csv is updated and
#     program exits
#     input:
#         root:  Root directory
#         subid: subject ID (without s prefix)
#         task: Task
#     output:
#       files: Dictionary with file paths (or empty lists).  Needs to be further
#           processed by check_file() to pick up instances when task is not available
#           for a given subject (missing data files)
#           Dictionary contains events_file, mask_file, confounds_file, data_file
#     """
#     files = {}
#     file_missing = {}
#     fmriprep_path = f"{root}/derivatives/fmriprep-24.1.0rc2"
#     file_missing["subid_task"] = f"{subid}_{task}"
#     files["events_file"], file_missing["event_file_missing"] = check_file(
#         glob.glob(f"{root}/sub-{subid}/func/*{task}*events.tsv")
#     )
#     files["confounds_file"], file_missing["confounds_file_missing"] = check_file(
#         glob.glob(
#             f"{fmriprep_path}/sub-{subid}/func/*{task}*confounds*.tsv"
#         )
#     )

#     files["mask_file"], file_missing["mask_file_missing"] = check_file(
#         glob.glob(
#             f"{fmriprep_path}/sub-{subid}/func/*{task}*MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
#         )
#     )
#     files["data_file"], file_missing["data_file_missing"] = check_file(
#         glob.glob(
#             f"{fmriprep_path}/sub-{subid}/func/*{task}*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#         )
#     )
#     print("Mask File =", files["mask_file"],file=sys.stderr)
#     print("Data File =", files["data_file"],file=sys.stderr)
#     file_missing = pd.DataFrame(file_missing)
    
#     if file_missing.loc[:, file_missing.columns != "subid_task"].gt(0).values.any():
#         update_excluded_subject_csv(file_missing, subid, task, contrast_dir)
#         print(f"Subject {subid}, task: {task} is missing one or more input data files.")
#         sys.exit(0)
#     return files
    
    
def get_files(root, subid, task, ses):  # Added ses as a parameter
    """Fetches files (events.tsv, confounds, mask, data) given a session
    if files are not present, excluded_subjects.csv is updated and
    program exits
    input:
        root:  Root directory
        subid: subject ID (without s prefix)
        task: Task
        ses: Session (e.g. 'ses-1' or 'ses-2')
    output:
      files: Dictionary with file paths (or empty lists).
    """
    files = {}
    file_missing = {}
    fmriprep_path = f"{root}/derivatives/fmriprep-24.1.0rc2"
    
    file_missing["subid_task"] = f"{subid}_{task}_{ses}"  # Include session in missing files tracking
    
    files["events_file"], file_missing["event_file_missing"] = check_file(
        glob.glob(f"{root}/sub-{subid}/{ses}/func/*{task}*events.tsv")
    )
    
    files["confounds_file"], file_missing["confounds_file_missing"] = check_file(
        glob.glob(f"{fmriprep_path}/sub-{subid}/{ses}/func/*{task}*confounds*.tsv")
    )
    
    
    files["mask_file"], file_missing["mask_file_missing"] = check_file(
        glob.glob(
            f"{fmriprep_path}/sub-{subid}/{ses}/func/*{task}*MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        )
    )

# Testing with 
#    files["mask_file"], file_missing["mask_file_missing"] = check_file(
#        glob.glob(
#            f"{fmriprep_path}/sub-{subid}/{ses}/func/*{task}*MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
#        )
#    )
 
    files["data_file"], file_missing["data_file_missing"] = check_file(
        glob.glob(
            f"{fmriprep_path}/sub-{subid}/{ses}/func/*{task}*MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        )
    )
    mask_pattern = f"{fmriprep_path}/sub-{subid}/{ses}/func/*{task}*MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    print(f"Searching for mask file with pattern: {mask_pattern}", file=sys.stderr)
    print("Mask File =", files["mask_file"], file=sys.stderr)
    print("Data File =", files["data_file"], file=sys.stderr)
    
    file_missing = pd.DataFrame(file_missing)
    
    if file_missing.loc[:, file_missing.columns != "subid_task"].gt(0).values.any():
        update_excluded_subject_csv(file_missing, subid, task, contrast_dir)
        print(f"Subject {subid}, task: {task}, session: {ses} is missing one or more input data files.")
        sys.exit(0)
        
    return files


def get_parser():
    """Build parser object"""
    parser = ArgumentParser(
        prog="analyze_lev1",
        description="analyze_lev1: Runs level 1 analyses with or without RT confound",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "task",
        choices=[
            "rest",
            "stopSignal",
            "discountFix",
            "motorSelectiveStop",
            "manipulationTask",
        ],
        help="Use to specify task.",
    )
    parser.add_argument(
        "subid",
        action="store",
        type=str,
        help="String indicating subject id",
    )
    
    parser.add_argument(
        "ses",
        action="store",
        type=str,
        choices=["ses-1", "ses-2"],  # You can expand the choices as needed
        help="Session identifier (e.g., ses-1 or ses-2).",
    )


    parser.add_argument(
        "regress_rt",
        choices=["no_rt", "rt_uncentered", "rt_centered"],
        help=(
            "Use to specify how rt is/is not modeled. If rt_centered is used "
            "the mean RT across all subjects is used for centering"
        ),
    )
    parser.add_argument(
        "--omit_deriv",
        action="store_true",
        help=(
            "Use to omit derivatives for task-related regressors "
            "(typically you would want derivatives)"
        ),
    )
    parser.add_argument(
        "--qa_only",
        action="store_true",
        help=(
            "Use this flag if you only want to QA model setup without estimating model."
        ),
    ),
    parser.add_argument(
        "--simplified_events",
        action="store_true",
        help=("Use this flag to output simplified event files."),
    )
    return parser


if __name__ == "__main__":
    from nilearn.glm.first_level import FirstLevelModel

    opts = get_parser().parse_args(sys.argv[1:])
    qa_only = opts.qa_only
    subid = opts.subid
    regress_rt = opts.regress_rt
    task = opts.task
    ses = opts.ses
    print("checkpoint1")
    simplified_events = opts.simplified_events
    if opts.omit_deriv:
        add_deriv = "deriv_no"
    else:
        add_deriv = "deriv_yes"
    outdir = (
        f"/oak/stanford/groups/russpold/data/uh2/aim4/BIDS/derivatives/"
        f"output/{task}_lev1_output/"
    )
    root = "/oak/stanford/groups/russpold/data/uh2/aim4/BIDS"
    #contrast_dir = f"{outdir}/task_{task}_rtmodel_{regress_rt}"
    contrast_dir = f"{outdir}/task_{task}_rtmodel_{regress_rt}/{ses}"  # Include session in output path

    if not os.path.exists(contrast_dir):
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(f"{contrast_dir}/contrast_estimates")

    files = get_files(root, subid, task, ses)
    #SMOOTHING HERE
    temp_path, temp_file_name = os.path.split(files["data_file"])
    temp_root, temp_root_ext = temp_file_name.split(".", maxsplit = 1)
    print("checkpoint2")
    smoothed = smooth_img(files["data_file"], fwhm = 5)
    output_smoothed_file = temp_path + "/" + temp_root + "_smooth" + "." + temp_root_ext
    smoothed.to_filename(output_smoothed_file)
    files["data_file"] = output_smoothed_file

    n_scans = get_nscans(files["data_file"])
    print("checkpoint3")

    (
        design_matrix,
        contrasts,
        percent_junk,
        percent_high_motion,
        tr,
        events_df,
    ) = make_desmat_contrasts(
        root,
        task,
        files["events_file"],
        add_deriv,
        n_scans,
        files["confounds_file"],
        regress_rt,
    )

    if simplified_events:
        if not os.path.exists(f"{contrast_dir}/simplified_events"):
            os.makedirs(f"{contrast_dir}/simplified_events")
        simplified_filename = f"{contrast_dir}/simplified_events/sub-{subid}_task-{task}_simplified-events.csv"
        events_df.to_csv(simplified_filename)
    print("checkpoint4")
    exclusion, any_fail = qa_design_matrix(
        contrast_dir,
        contrasts,
        design_matrix,
        subid,
        task,
        percent_junk,
        percent_high_motion,
    )
    print("checkpoint5")
    add_to_html_summary(
        subid,
        contrasts,
        design_matrix,
        contrast_dir,
        regress_rt,
        task,
        any_fail,
        exclusion,
    )
    print("checkpoint6")
    if not any_fail and qa_only == False:
        fmri_glm = FirstLevelModel(tr,
                                    subject_label=subid,
                                    mask_img=files['mask_file'],
                                    noise_model='ar1',
                                    standardize=False,
                                    drift_model=None
                                    )
        out = fmri_glm.fit(files['data_file'], design_matrices = design_matrix)
        print("checkpoint7")
        for con_name, con in contrasts.items():
            filename = (f'{contrast_dir}/contrast_estimates/task_{task}_contrast_{con_name}_sub_'
                 f'{subid}_rtmodel_{regress_rt}_stat'
                 f'_contrast.nii.gz')
            con_est  = out.compute_contrast(con, output_type = 'effect_size')
            con_est.to_filename(filename)
            
    print("checkpoint7")