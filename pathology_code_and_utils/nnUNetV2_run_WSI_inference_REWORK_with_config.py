#################################################################
### NNUNET WSI INFERENCE WITH HALF OVERLAP
#################################################################
### Joey Spronck
# nnUNet by default does half overlap if the given patch is bigger than the model's patch size.
# This means that on the borders there is no or 1x overlap (1 or 2 predictions),
# while in the inside there are 4 predictions for each pixel.

# In this version of nnUNet WSI inference, we crop this border off the sampled patch, and only write the inner part
# This approach becomes more efficient if bigger patches are sampled, because this increases the inner/outer ratio
# To prevent inference on large empty patches we add a check if we can remove rows and columns,
# while preserving the half overlap of nnUNet's sliding window approach

# In this pipeline I simplified the sampling method a lot
#################################################################

#################################################################
### Imports
#################################################################

import os
import sys
import shutil
import importlib
from pathlib import Path
import numpy as np
import torch
import time
import wandb

from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators.patchiterator import create_patch_iterator
from wholeslidedata.buffer.patchcommander import PatchConfiguration
from wholeslidedata.interoperability.asap.imagewriter import WholeSlideMaskWriter
from wholeslidedata.samplers.utils import crop_data

from nnunetv2.utilities.file_path_utilities import load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

#################################################################
### CONFIG IMPORT
#################################################################

def import_config(identifier):
    """
    Load a config module either by:
      1. Full path to a .py file
      2. Config name (no slashes), loaded relatively from inference_configs/
    
    Both paths ultimately use importlib.import_module, so they return
    the same module object if the file/module is the same.
    """
    identifier_path = Path(identifier)
    
    # --- Decide base directory and module name ---
    if identifier_path.suffix == ".py" or identifier_path.is_absolute() or len(identifier_path.parts) > 1:
        # Option 1: Full path
        config_path = identifier_path.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        base_dir = str(config_path.parent)
        module_name = config_path.stem
    else:
        # Option 2: Config name
        base_dir = str(Path(__file__).resolve().parent / "inference_configs")
        module_name = identifier

    # --- Ensure the base directory is in sys.path ---
    if base_dir not in sys.path:
        sys.path.append(base_dir)

    # --- Import the module ---
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Config module '{module_name}' not found in '{base_dir}'"
        )

if len(sys.argv) != 2 and len(sys.argv) != 8:
    print("\n\n\nINCORRECT FUNCTION CALL: \nPlease provide a config stem as argument and optionally the input and output paths")
    print("\n[Default config usage]: \n\tpython nnUNet_run_WSI_inference_REWORK_using_config.py <config_stem>")
    print('\nOR')
    print("\n[Config usage + providing in and output paths]: \n\tpython nnUNet_run_WSI_inference_REWORK_using_config.py <config_stem> <input_img> <input_mask> <output_inference_mask> <output_uncertainty_CE_mask> <output_uncertainty_KL_mask> <output_uncertainty_entropy_mask>")
    sys.exit(1)

config_module_name = sys.argv[1]
config = import_config(config_module_name)

### TASK AND MODEL STUFF'
model_base_path = config.model_base_path

norm = config.norm 
output_minus_1 = config.output_minus_1

### OUTPUT FOLDER
output_img, output_mask = (None, None) if len(sys.argv) == 2 else (sys.argv[4], sys.argv[5])
if output_img is not None and output_mask is not None:
    # Should have same parent
    assert Path(output_img).parent == Path(output_mask).parent, "Output paths should have the same parent, since this is where the runtime files are stored"
output_folder = config.output_folder if len(sys.argv) == 2 else Path(output_img).parent
local_output_folder = Path('/tmp/workdir')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(local_output_folder, exist_ok=True)

### MATCHES YOU WANT TO RUN
# This is a list of tuples, where each tuple contains the path to the wsi and the path to the tissue mask
# Example: [('path/to/wsi1', 'path/to/tissue_mask1'), ('path/to/wsi2', 'path/to/tissue_mask2')]
matches_to_run = config.matches_to_run if len(sys.argv) == 2 else [(sys.argv[2], sys.argv[3])]

rerun_unfinished = config.rerun_unfinished # [READ THIS -->] this will check if the '<stem>_runtime.txt file is present in the outputfolder. If it is not it means that this file is not processed fully. IMPORTANT: if this is set to True multiple parallel jobs cannot track which slides are being processed, set this to false if you want to run multiple jobs in a parralel way. Alternatively, remove all unfinished files in the output folder (the files that do nt have a .runtime.txt file). Default is False 
overwrite = config.overwrite # [READ THIS -->] this will always process and overwrite files, even if they are already processed. May break logic if you have multiple jobs running in parallel. Default is False

### SAMPLING STUFF
spacing = config.spacing # spacing for batch_shape spacing and annotation_parser output_spacing (leave its processing spacing on 4 or higher)
model_patch_size = config.model_patch_size # input size of model (should be square)
sampler_patch_size = config.sampler_patch_size # use this as batch shape (= 8 * model_patch_size)
cpus = getattr(config, 'cpus', 1) # number of cpus to use for sampling (note that buffer states are printed during inference time now, which may give you information about the number of cpus you should use, I noticed 1 is already enough to have a saturated buffer state when you sample: 4 * model_patch_size)

### WANDB
use_wandb = config.use_wandb
if use_wandb:
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key is None:
        print("WANDB_API_KEY not found in environment variables. Please put 'export WANDB_API_KEY=<your key>' in your .bashrc or .bash_profile")
        print("Aborting...")
        sys.exit(1) 

#################################################################
### Functions and utilities
#################################################################
current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"

def convert_path(path, to=current_os):
    '''
    This function converts paths to the format of the desired platform. 
    By default it changes paths to the platform you are currently using.
    This way you do not need to change your paths if you are executing your code on a different platform.
    It may however be that you mounted the drives differently. 
    In that case you may need to change that in the code below. 
    For example, change the Z for blissey (where I mounted it) to X (how you may have mounted it)
    If you add a linux path on a linux platform, or windows on windows, this function by default doesnt do anything.
    '''
    if to in ["w", "win", "windows"]:
        path = path.replace("/data/pathology", "Z:")
        path = path.replace("/data/pa_cpgarchive1", "W:")
        path = path.replace("/data/pa_cpgarchive2", "X:")
        path = path.replace("/data/pa_cpg", "Y:")
        path = path.replace("/data/temporary", "T:")
        path = path.replace("/", "\\")
    if to in ["u", "unix", "l", "linux"]:
        path = path.replace("Z:", "/data/pathology")
        path = path.replace("W:", "/data/pa_cpgarchive1")
        path = path.replace("X:", "/data/pa_cpgarchive2")
        path = path.replace("Y:", "/data/pa_cpg")
        path = path.replace("T:", "/data/temporary")
        path = path.replace("\\", "/")
    return path

def ensemble_softmax_list(x_batch):
    logits_list = predictor.get_logits_list_from_preprocessed_data(torch.tensor(x_batch, dtype=torch.float32))
    softmax_list = [predictor.label_manager.apply_inference_nonlin(logits).numpy() for logits in logits_list]
    return softmax_list

def array_to_formatted_tensor(array):
    # array = np.expand_dims(array.transpose(2, 0, 1), 0)
    array = array.transpose(1, 0, 2, 3)
    return torch.tensor(array) # need (1, classes, w/h, h/w)

def softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    uncertainty_loss_per_pixel_list = []
    for softmax in softmax_list:
        log_softmax = np.log(softmax + 0.00000001)
        uncertainty_loss_per_pixel = loss(array_to_formatted_tensor(log_softmax),
                                          array_to_formatted_tensor(softmax_mean))
        uncertainty_loss_per_pixel_list.append(uncertainty_loss_per_pixel)
    uncertainty = torch.cat(uncertainty_loss_per_pixel_list).mean(dim=0)
    return uncertainty

def get_trim_indexes(y_batch):
    """
    Using the y_mask / tissue-background mask we can check if there are
    full empty rows and columns with a width of half the model patch size.
    We check this in half model patch size increments because otherwise
    we screw up the half overlap approach from nnunet (resulting in inconsistent
    overlap thoughout the WSI).
    We will still need 1 row or column that is empty to make sure the parts that
    do have tissue have 4x overlap
    """
    y = y_batch[0]
    r_is_empty = [not y[start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]
    c_is_empty = [not y[:, start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]

    empty_rs_top = 0
    for r in r_is_empty:
        if r == True:
            empty_rs_top += 1  # count empty rows
        else:
            trim_top_half_idx = empty_rs_top - 1  # should always include a single empty row, since we need the overlap
            trim_top_half_idx = np.clip(trim_top_half_idx, 0, None)  # cannot select regiouns outside sampled patch
            trim_top_idx = half_patch_size_start_idxs[trim_top_half_idx]
            break

    empty_rs_bottom = 0
    for r in r_is_empty[::-1]:
        if r == True:
            empty_rs_bottom += 1
        else:
            trim_bottom_half_idx = empty_rs_bottom - 1
            trim_bottom_half_idx = np.clip(trim_bottom_half_idx, 0, None)
            trim_bottom_idx = half_patch_size_end_idxs[::-1][trim_bottom_half_idx]  # reverse index
            break

    empty_cs_left = 0
    for c in c_is_empty:
        if c == True:
            empty_cs_left += 1
        else:
            trim_left_half_idx = empty_cs_left - 1
            trim_left_half_idx = np.clip(trim_left_half_idx, 0, None)
            trim_left_idx = half_patch_size_start_idxs[trim_left_half_idx]
            break

    empty_cs_right = 0
    for c in c_is_empty[::-1]:
        if c == True:
            empty_cs_right += 1
        else:
            trim_right_half_idx = empty_cs_right - 1
            trim_right_half_idx = np.clip(trim_right_half_idx, 0, None)
            trim_right_idx = half_patch_size_end_idxs[::-1][trim_right_half_idx]
            break

    return trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx

def decode_buffer_states(state_array, cpus):
    state_mappings = {
        'FREE': 1,
        'AVAILABLE': 2,
        'RESERVED': 3,
        'PROCESSING': 4
    }
    state_count = {state: np.sum(state_array == state_mappings[state]) for state in state_mappings}
    sum_states = sum(state_count.values())
    if state_count['AVAILABLE'] + state_count['PROCESSING'] == sum_states:
        message = f', your iterator buffer is saturated.\n\t\tIf you see this all the time you probably dont need this many CPUs for the processing_iterator. Currently using: {cpus} CPUs'
    elif (state_count['AVAILABLE'] == 0) or (state_count['AVAILABLE'] == 1):
        message = f', your iterator buffer is empty or almost empty.\n\t\tIf you see this all the time you may benifit from using more CPUs for the processing_iterator. Currently using: {cpus} CPUs'
    else: 
        message = ''
    return state_count, message

def get_closest_value(value):
    possible_values = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    closest = min(possible_values, key=lambda x:abs(x-value))
    return closest

def asap_validation(path):
    try:
        import multiresolutionimageinterface as mir
        reader = mir.MultiResolutionImageReader()
        print(f"ASAP validation of {path}")
        wsi = reader.open(str(path))
        if wsi is None:
            return False
        else:
            return wsi.valid()
    except Exception as e:
        print(f"ASAP validation error: {e}")
        return False

#################################################################
### LOAD MODEL
#################################################################
print('\nModel path:')
print(model_base_path, '\n')

plans_dict = load_json(os.path.join(model_base_path, 'plans.json'))
dataset_dict = load_json(os.path.join(model_base_path, 'dataset.json'))

# trainer = nnUNetTrainer_custom_dataloader_test(plans_dict, '2d', 0, dataset_dict) # we need a trainer for a single fold to make use of its inbuilt functions

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_gpu=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False
)

predictor.initialize_from_trained_model_folder(
    model_base_path,
    use_folds=(0, 1, 2, 3, 4),
    checkpoint_name='checkpoint_best.pth',
)

#################################################################
### AUTO CONFIG
#################################################################
half_model_patch_size=model_patch_size//2
assert sampler_patch_size % half_model_patch_size == 0 # needed for correct half overlap
# due to half overlap there is half the model patch size without overlap on all 4 sides of the sampled patch
output_patch_size = sampler_patch_size - 2 * half_model_patch_size # use this as your annotation_parser shape

# following is later used to check if we can remove big empty parts of the sampled patch before inference
sampler_patch_size_range = list(range(sampler_patch_size))
half_patch_size_start_idxs = sampler_patch_size_range[0::half_model_patch_size]
half_patch_size_end_idxs = [idx + half_model_patch_size for idx in half_patch_size_start_idxs]
wsm_writer = None

#################################################################
### WANDB
#################################################################
if use_wandb:
    import datetime
    print('Wandb init')
    date = datetime.datetime.now().strftime("%Y%m%d")

    wandb.login(key=wandb_api_key)
    wandb.init(
        project=f'nnUNet_inference_checks', 
        name = f'sample: {sampler_patch_size} patch: {model_patch_size} date: {date}'
        )

##################################################################################################################################
## LOOP
##################################################################################################################################
print('\n\n\n####### START OF LOOP #######')
for idx_match, (image_path, mask_path) in enumerate(matches_to_run):
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    print(f'\n[NEXT MATCH] [{idx_match}/{len(matches_to_run)})]:', '\n\t', image_path, mask_path)
    
    ### CHECK IF WE NEED TO PROCESS THIS FILE
    wsm_path = output_folder / (image_path.stem + '_nnunet.tif')
    wsu_path = output_folder / (image_path.stem + '_uncertainty.tif')
    if not overwrite:
        if os.path.isfile(output_folder / (image_path.stem + '_runtime.txt')):
            print(f'[SKIPPING] {image_path.stem} is processed already', flush=True)
            continue  # continue to next match
        if not rerun_unfinished:
            if os.path.isfile(wsm_path) and os.path.isfile(wsu_path):
                print(f'[SKIPPING] {image_path.stem} is processed already or currently being processed', flush=True)
                continue  # continue to next match
    # Immediately lock files
    open(wsm_path, 'w').close()
    open(wsu_path, 'w').close()

    print(f'[RUNNING] {image_path.stem}\n', flush=True)

    ### LOAD WSI AND GET SAMPLER/WRITER SETTINGS
    with WholeSlideImage(image_path, backend='asap') as wsi:
        shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
        real_spacing = wsi.get_real_spacing(spacing)
        downsampling = get_closest_value(wsi.get_downsampling_from_spacing(spacing))
        offset = int((output_patch_size // 2) * downsampling) # this was difficult...

    patch_configuration = PatchConfiguration(patch_shape=(sampler_patch_size,sampler_patch_size,3),
                                            spacings=(spacing,),
                                            overlap=(model_patch_size,model_patch_size),
                                            offset=(int(offset), int(offset)),
                                            center=True,
                                            write_shape=(output_patch_size, output_patch_size))

    # Create new writer and file
    start_time = time.time()
    wsm_writer = WholeSlideMaskWriter()  # whole slide mask
    wsu_writer = WholeSlideMaskWriter()  # whole slide uncertainty
    # Create files
    wsm_path_local = local_output_folder / (image_path.stem + '_nnunet.tif')
    wsu_path_local = local_output_folder / (image_path.stem + '_uncertainty.tif')
    wsm_writer.write(path=wsm_path_local, spacing=real_spacing, dimensions=shape,
                    tile_shape=(output_patch_size, output_patch_size))
    wsu_writer.write(path=wsu_path_local, spacing=real_spacing,
                    dimensions=shape, tile_shape=(output_patch_size, output_patch_size))

    #################################################################
    ### RUN
    #################################################################
    print('\nInitiating iterator...')

    with create_patch_iterator(image_path=image_path,
        mask_path=mask_path,
        patch_configuration=patch_configuration,
        backend='asap',
        cpus=4) as patch_iterator:

        print('Iterator initiated', flush=True)
        print('Starting inference...\n\n', flush=True)

            # time calling next start
        time_pre_next = time.time()

        for idx_batch, (x_batch, y_batch, info) in enumerate(patch_iterator):
            time_post_next = time.time()
            duration_next = time_post_next - time_pre_next
                # time calling next end

                # time patch processing start
            time_pre_patch_processing = time.time()

            ### Print progress
            # if idx_batch%10==0: 
            print(f'\t[processing tile {idx_batch}/{len(patch_iterator)}] ...', flush=True)

            
            ### Trim check and prep
                # time prep start
            time_pre_prep = time.time()
            x_batch = x_batch[0]
            y_batch = y_batch[0]
            trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx = get_trim_indexes(y_batch)
            x_batch_maybe_trimmed = x_batch[:, trim_top_idx : trim_bottom_idx, trim_left_idx: trim_right_idx, :]
            prep = norm(x_batch_maybe_trimmed)
            time_post_prep = time.time()
            duration_prep = time_post_prep - time_pre_prep
                # time prep end
            ###

            ### Predict and uncertainty
                # time predict start
            time_pre_predict = time.time()
            softmax_list = ensemble_softmax_list(prep)
            softmax_mean = np.array(softmax_list).mean(0)
            pred_output_maybe_trimmed = softmax_mean.argmax(axis=0)-(1 if output_minus_1 else 0)
            time_post_predict = time.time()
            duration_predict = time_post_predict - time_pre_predict
                # time predict end
            ###

            ### Uncertainty
                # time uncertainty start
            time_pre_uncertainty = time.time()
            uncertainty = softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean)
            uncertainty_output_maybe_trimmed = np.array((uncertainty.clip(0, 4) / 4 * 255).int()) 
            time_post_uncertainty = time.time()
            duration_uncertainty = time_post_uncertainty - time_pre_uncertainty
                # time uncertainty end
            ###

            ### Patch wrangling and writing
                # time patch wrangling start
            time_pre_patch_wrangling = time.time()
            # Reconstruct possible trim
            pred_output = np.zeros((sampler_patch_size, sampler_patch_size))
            pred_output[trim_top_idx : trim_bottom_idx, trim_left_idx: trim_right_idx] = pred_output_maybe_trimmed
            uncertainty_output = np.zeros((sampler_patch_size, sampler_patch_size))
            uncertainty_output[trim_top_idx: trim_bottom_idx, trim_left_idx: trim_right_idx] = uncertainty_output_maybe_trimmed
            # Only write inner part
            pred_output_inner = crop_data(pred_output, [output_patch_size, output_patch_size])
            uncertainty_output_inner = crop_data(uncertainty_output, [output_patch_size, output_patch_size])
            y_batch_inner = crop_data(y_batch[0], [output_patch_size, output_patch_size]).astype('int64')
            # Get patch point
            x_coord, y_coord = info['x']//downsampling, info['y']//downsampling # convert coordinates to writing spacing
            x_coord, y_coord = x_coord - output_patch_size/2, y_coord - output_patch_size/2 # from middle point to upper left point of tile to write
            time_post_patch_wrangling = time.time()
            duration_patch_wrangling = time_post_patch_wrangling - time_pre_patch_wrangling
                # time patch wrangling end
            ###

            ### Write tile
                # time writing start
            time_pre_writing = time.time()
            wsm_writer.write_tile(tile=pred_output_inner * y_batch_inner, coordinates=(int(x_coord), int(y_coord)))
            wsu_writer.write_tile(tile=uncertainty_output_inner * y_batch_inner, coordinates=(int(x_coord), int(y_coord)))
            time_post_writing = time.time()
            duration_writing = time_post_writing - time_pre_writing
                # time writing end
            ###
    
            time_post_patch_processing = time.time()
            duration_patch_processing = time_post_patch_processing - time_pre_patch_processing
                # time patch processing end

            if use_wandb:
                print("\t\t[LOGGING] to wandb")
                wandb.log({
                    "duration_next": duration_next,
                    "NORM_duration_next": duration_next / (output_patch_size**2) ,
                    "duration_prep": duration_prep,
                    "NORM_duration_prep": duration_prep / (output_patch_size**2) ,
                    "duration_predict": duration_predict,
                    "NORM_duration_predict": duration_predict / (output_patch_size**2) ,
                    "duration_uncertainty": duration_uncertainty,
                    "NORM_duration_uncertainty": duration_uncertainty / (output_patch_size**2) ,
                    "duration_patch_wrangling": duration_patch_wrangling,
                    "NORM_duration_patch_wrangling": duration_patch_wrangling / (output_patch_size**2),
                    "duration_writing": duration_writing,
                    "NORM_duration_writing": duration_writing / (output_patch_size**2),
                    "duration_patch_processing": duration_patch_processing,
                    "NORM_duration_patch_processing": duration_patch_processing / (output_patch_size**2),
                    "duration_total": time_post_writing - time_pre_next,
                    "NORM_duration_total": (time_post_writing - time_pre_next) / (output_patch_size**2) 
                })
                
            if idx_batch%10==0: 
                state_array = patch_iterator._buffer_factory.buffer_state_memory.get_state_buffer()
                state_count, message = decode_buffer_states(state_array, cpus)
                print(f'\t\tBUFFER STATES (batch {idx_batch}): {state_count}, {message}', flush=True)

                # time calling next start
            time_pre_next = time.time()
        print('[PROCESSED ALL TILES]\n\n', flush=True)

    # Save runtime
    end_time = time.time()
    run_time = end_time - start_time
    text_file = open(output_folder / (image_path.stem + '_runtime.txt'), "w")
    text_file.write(str(run_time))
    text_file.close()

    print('[WRITING, VALIDATING, and TRANSFERING] inference and uncertainty masks\n', flush=True)
    wsm_writer.save()  # if done save image
    print(f'Verification of written inference:', wsm_path_local, flush=True)
    if asap_validation(wsm_path_local):
        print('\tVerification successful', flush=True)
        print(f'\tCopying {wsm_path_local} to {wsm_path}\n\n', flush=True) 
        shutil.copyfile(wsm_path_local, wsm_path)
    else:
        print('\tVerification failed', flush=True)
        print(f'\tRemoving initialized {wsm_path} and {wsu_path}', flush=True)
        os.remove(wsm_path)
        os.remove(wsu_path)
        os.remove(text_file)
        sys.exit(1)

    wsu_writer.save()  # if done save image
    if asap_validation(wsu_path_local):
        print('\tVerification successful', flush=True)
        print(f'\tCopying {wsu_path_local} to {wsu_path}\n\n', flush=True) 
        shutil.copyfile(wsu_path_local, wsu_path)
    else:
        print('\tVerification failed', flush=True)
        print(f'\tRemoving initialized {wsm_path} and {wsu_path}', flush=True)
        os.remove(wsm_path)
        os.remove(wsu_path)
        os.remove(text_file)
        sys.exit(1)

    patch_iterator.stop()

if wsm_writer == None:
    print('\n\n[NO FILES TO PROCESS] \n\n\n', flush=True)

print('[COMPLETELY DONE]\nIf there are remaining files that are not being processed right now, set "rerun_unfinished" to True in your config ### RUN section')
print('\n\n\n[Potential incoming multiprocessing error] still exiting with exit status 0\n\n')

# Exit successfully
sys.exit(0)