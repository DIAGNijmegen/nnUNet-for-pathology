#!/usr/local/bin/python3

import argparse
import os
import pickle
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
from nnunet.utilities import shutil_sol
import hashlib
import json

PLANS = 'nnUNetPlansv2.1'

PathLike = Union[str, Path]

def get_task_id(task_name):
    return re.match('Task([0-9]+)', task_name).group(1)

def copy_image(srcfile, dstfile):
    if srcfile.name.endswith('.nii.gz'):
        shutil_sol.copyfile(srcfile, dstfile)
    else:
        # Not supported for pathology
        raise RuntimeError(f'Unsupported image format: {srcfile.name}, should be .nii.gz')
    
def checksum(file: PathLike, algorithm: str = "sha256", chunk_size: int = 4096) -> str:
    """Computes the checksum of a file using a hashing algorithm"""
    file = Path(file)
    if file.exists() and not file.is_file():
        raise ValueError(f"Checksum can be computed only for files, {file} is not a file")

    h = hashlib.new(algorithm)
    with file.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def refresh_file_list(path: PathLike):
    """Update the cached file list of directories on a network share

    On network shares (chansey in particular), files are sometimes reported missing even though they
    exist. This has to do with caching issues and can be fixed by running "ls" or similar commands
    in the parent directory of these files.
    """
    path = Path(path)  # make sure path is a OS-specific path object
    if sys.platform == "win32":
        cmd = ["cmd", "/c", "dir", str(path)]
    else:
        cmd = ["ls", str(path)]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise OSError(f'Could not refresh file list of directory "{path}"') from e

def path_exists(path: PathLike) -> bool:
    """Checks whether the file or directory exists

    Unlike checks via os.path or pathlib, this check works reliably also on network shares where the
    content of directories might be cached.
    """
    path = Path(path)

    # Refresh content of parent (folder in which the object of interest is stored)
    try:
        refresh_file_list(path.parent)
    except OSError:
        # If the parent directory does not exist, refreshing the file list will fail, but that's okay
        pass

    # Now we can check if the specific file/directory exists
    if path.exists():
        # If the object was a directory itself, we also ask for a fresh list of it's content
        if path.is_dir() and path != path.parent:
            refresh_file_list(path)
        return True
    else:
        return False

def read_json(filename: PathLike, *, ordered_dict: bool = True, **kwargs):
    """Reads a json file"""
    if ordered_dict:
        kwargs["object_pairs_hook"] = OrderedDict

    with Path(filename).open() as fp:
        return json.load(fp, **kwargs)

def plan_train(argv):
    # Plan experiment, then train network
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--results', type=str, required=False)
    parser.add_argument('--network', type=str, default='2d') # Changed default to 2d for pathology
    parser.add_argument('--trainer', type=str, default='nnUNetTrainerV2')
    parser.add_argument('--trainer_kwargs', required=False, default="{}",
                        help="Use a dictionary in string format to specify keyword arguments. This will get"
                             " parsed into a dictionary, the values get correctly parsed to the data format"
                             " and passed to the trainer. Example (backslash included): \n"
                             r"--trainer_kwargs='{\"class_weights\":[0,2.00990337,1.42540704,2.13387239,0.85529504,0.592059,0.30040984,8.26874351],\"weight_dc\":0.3,\"weight_ce\":0.7}'")
    parser.add_argument("--plans", type=str, default="nnUNetPlansv2.1", help="Can be used to specify a custom identifier for the plans file")
    parser.add_argument("--planner3d", type=str, default="ExperimentPlanner3D_v21",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                        "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                        "configured",)
    parser.add_argument("--planner2d", type=str, default="None", 
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is 'None', so that the 2D U-Net is not configured, saving time during planning"
                        "2D nnUNet default planner is 'ExperimentPlanner2D_v21'.",)
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--custom_split', type=str, help='Path to a JSON file with a custom data split into five folds')
    parser.add_argument('--plan_only', action='store_true', help='Run the planning step, but not the training step')
    parser.add_argument('--validation_only', action='store_true',
                        help='Do no run network training, only the final validation step')
    parser.add_argument('--ensembling', action='store_true',
                        help='Export probability maps for ensembling during the final validation')
    parser.add_argument('--use_compressed_data', action='store_true',
                        help='Disable unpacking of compressed training data, use with caution')
    parser.add_argument('--carbontracker', action='store_true', help='Enables tracking of energy consumption')
    parser.add_argument('--pretrained_weights', type=str, required=False, default=None)
    args = parser.parse_args(argv)

    # Set environment variables
    datadir = Path(args.data)
    prepdir = Path('/home/user/data')
    splits_file = prepdir / args.task / 'splits_final.pkl'

    os.environ['nnUNet_raw_data_base'] = str(datadir)
    os.environ['nnUNet_preprocessed'] = str(prepdir)
    os.environ['RESULTS_FOLDER'] = args.results if args.results else str(datadir / 'results')

    # Check if plans and preprocessed data are available
    taskid = get_task_id(args.task)
    taskdir = datadir / 'nnUNet_preprocessed' / args.task

    if path_exists(taskdir / args.plans):
        if args.custom_split:
            remote_splits_file = taskdir / 'splits_final.json'
            if not remote_splits_file.exists() or checksum(remote_splits_file) != checksum(args.custom_split):
                print(f"[#] Found plans and preprocessed data for {args.task}"
                        " - but you also provided a custom split which is different"
                        " from the present split, this is not permitted")
                return

        if args.plan_only:
            print(f'[#] Found plans and preprocessed data for {args.task} - nothing to do')
        else:
            print(f'[#] Found plans and preprocessed data for {args.task} - copying to compute node')
            if not os.path.exists(prepdir / args.task):
                prepdir.mkdir(parents=True, exist_ok=True)
                shutil_sol.copytree(taskdir, prepdir / args.task)
            print(f'[#] Found plans and preprocessed data for {args.task} - copied to compute node')
    else:
        # Plans and data not available yet, run preprocessing
        print('[#] Creating plans and preprocessing data')
        cmd = [
            'nnUNet_plan_and_preprocess',
            '-t', taskid,
            '-tl', '1', '-tf', '1',
            '--verify_dataset_integrity'
        ]
        if args.planner2d == "None" and '2d' not in args.network:
            cmd.extend(['--planner2d', 'None'])  # disable 2D planning to speed up the preprocessing phase
        else:
            cmd.extend(["-pl2d", args.planner2d]) # Running with specified 2D planner
        if args.planner2d == "None" and '3d' not in args.network:
            cmd.extend(['--planner3d', 'None'])
        else:
            cmd.extend(["-pl3d", args.planner3d]) # Running with specified 3D planner
        if args.pretrained_weights is not None:
            cmd.extend(['-pretrained_weights', args.pretrained_weights])
            
        subprocess.check_call(cmd)

        # Use a custom data split?
        if args.custom_split:
            splits = []
            for split in read_json(args.custom_split):
                splits.append(OrderedDict([
                    ('train', np.array(split['train'])),
                    ('val', np.array(split['val']))
                ]))

            splits_file.parent.mkdir(parents=True, exist_ok=True)
            with splits_file.open('wb') as fp:
                pickle.dump(splits, fp)
            shutil_sol.copyfile(args.custom_split, splits_file.with_suffix('.json'))

        # Copy preprocessed data to storage server
        print('[#] Copying plans and preprocessed data from compute node to storage server')
        taskdir.parent.mkdir(parents=True, exist_ok=True)
        shutil_sol.copytree(prepdir / args.task, taskdir, dirs_exist_ok=True)

    if args.plan_only:
        return

    # Run training
    cmd = [
        'nnUNet_train',
        args.network,
        args.trainer,
        taskid,
        args.fold
    ]

    fold_name = 'all' if args.fold == 'all' else f'fold_{args.fold}'
    outdir = Path(
        os.environ['RESULTS_FOLDER']) / 'nnUNet' / args.network / args.task / f'{args.trainer}__{args.plans}' / fold_name

    if args.validation_only:
        print('[#] Running validation step only')
        cmd.append('--validation_only')
    elif path_exists(outdir) and any(outdir.glob("*.model")):
        print('[#] Resuming network training')
        cmd.append('-c')
    else:
        print('[#] Starting network training')

    if args.trainer_kwargs:
        cmd.append(fr'--trainer_kwargs=%s' % args.trainer_kwargs)
    if args.use_compressed_data:
        cmd.append('--use_compressed_data')
    if args.ensembling or args.network == '3d_lowres':
        cmd.append('--npz')
    cmd.extend(['-p', args.plans])


    subprocess.check_call(cmd)

    # Copy split file since that is for sure available now (nnUNet_train has created
    # it the file did not exist already - unless training with "all", so still check)
    if splits_file.exists():
        shutil_sol.copyfile(splits_file, taskdir)

def checkout(argv):
    # Switch to a specific branch of the nnU-Net repository?
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=str, default='')
    args, unknown = parser.parse_known_args(argv)

    if args.checkout:
        subprocess.check_call([
            'git', '-C', '/home/user/nnunet',
            'fetch'
        ])
        subprocess.check_call([
            'git', '-C', '/home/user/nnunet',
            'checkout', args.checkout
        ])
    return unknown


if __name__ == '__main__':
    # Very first argument determines action, note that all actions other than 'plan_train' were removed for pathology
    actions = {
        'plan_train': plan_train,
    }

    try:
        action = actions[sys.argv[1]]
        argv = checkout(sys.argv[2:])
    except (IndexError, KeyError):
        print('Usage: nnunet ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(argv)