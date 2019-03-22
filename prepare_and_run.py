import glob
import os
import shutil
import sys
import argparse
import datetime
import re
from pathlib import Path
import yaml
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--auto_inference_dir', help="Auto Annotation directory.", type=str, required=False, default=None)
parser.add_argument('-i', '--inference_dir', help="Inferences destination", type=str, required=True)
parser.add_argument('-w', '--framework', help="keras-yolo3 framework path", type=str, required=True)
parser.add_argument('-p', '--min_overlap', help="List of minimum overlaps which will be tested individually.", required=True, default=None, nargs='+')
parser.add_argument('-f', '--force_overwrite', help="Overwrite already generated resources.", action="store_true")
parser.add_argument("-r", "--run_inferences", required=False, action="store_true", help="Run inferences over all directories in Inferences destination folder?")
parser.add_argument("-ca", "--canonical_bboxes", required=False, action="store_true", help="The training configuration.")
main_args = parser.parse_args()

if main_args.canonical_bboxes:
    raise Exception('Canonical bbox are not supported through this script. Run directly with eval_all_inferences.py')

destination_root_path = os.path.join(main_args.inference_dir, 'auto_results')
os.makedirs(destination_root_path,exist_ok=True)

inferences = glob.glob(os.path.join(main_args.auto_inference_dir, 'infer_*.txt'))
inferences.sort()

print('We have found {} inference files.'.format(len(inferences)))

epoch_regex = re.compile('_(ep[0-9]{3})|(trained_weights_final.h5)_')


exp_regex_compilation_patterns = []
tiny_infusion_pattern = '_logsseg-'
exp_regex_compilation_patterns.append(tiny_infusion_pattern)
tiny_pattern = 'logsdefault-tiny-'
exp_regex_compilation_patterns.append(tiny_pattern)
default_pattern = 'logsdefault-'
exp_regex_compilation_patterns.append(default_pattern)
default_infusion_pattern = '_logsdefault-seg-'
exp_regex_compilation_patterns.append(default_infusion_pattern)

caltech_yolov3_tiny = '_logscaltech_default-'
exp_regex_compilation_patterns.append(caltech_yolov3_tiny)
caltech_yolov3 = '_logscaltech_yolov3-'
exp_regex_compilation_patterns.append(caltech_yolov3)
caltech_yolov3_tiny_infusion = '_logscaltech_seg-'
exp_regex_compilation_patterns.append(caltech_yolov3_tiny_infusion)
caltech_yolov3_infusion = '_logscaltech_yolov3-seg-'
exp_regex_compilation_patterns.append(caltech_yolov3_infusion)

exp_regex_compilation_string = ''
for i, exp_pattern in enumerate(exp_regex_compilation_patterns):
    if i != 0:
        exp_regex_compilation_string += '|' #add or statement

    exp_regex_compilation_string += '('+exp_pattern+')[0-9]{3}'
# print(exp_regex_compilation_string)
experiment_number_regex = re.compile(exp_regex_compilation_string)

if main_args.auto_inference_dir:
    for inf in inferences:
        epoch_match = epoch_regex.search(inf)
        if epoch_match:
            epoch_match = epoch_match.group()
            destination_folder = os.path.basename(inf).replace('.txt','').replace(epoch_match,'')
            destination_path = os.path.join(destination_root_path,destination_folder)
            os.makedirs(destination_path,exist_ok=True)
            auto_gen_mark = os.path.join(destination_path,'this_have_been_auto_generated')
            Path(auto_gen_mark).touch()
            shutil.move(inf, destination_path)
        else:
            print(inf)
            raise Exception("Could not find the epoch number in the inference file name.")

if main_args.run_inferences:
    # import eval_all_inferences

    for inf_folder in os.listdir(destination_root_path):
        config_file = None
        print(inf_folder)
        experiment_match = experiment_number_regex.search(inf_folder)
        if experiment_match:
            experiment_match = experiment_match.group()
            print(experiment_match)
            if tiny_infusion_pattern in experiment_match:
                config_file = 'train-pti01-yolov3-tiny-infusion-config-seg-{}.yml'.format(experiment_match.replace(tiny_infusion_pattern,''))
            elif default_infusion_pattern in experiment_match:
                config_file = 'train-pti01-yolov3-infusion-config-{}.yml'.format(experiment_match.replace(default_infusion_pattern,''))
            elif default_pattern in experiment_match and not 'tiny' in experiment_match:
                config_file = 'train-pti01-yolov3-config-{}.yml'.format(experiment_match.replace(default_pattern,''))
            elif tiny_pattern in experiment_match:
                config_file = 'train-pti01-yolov3-tiny-config-{}.yml'.format(experiment_match.replace(tiny_pattern,''))
            elif caltech_yolov3_tiny in experiment_match:
                config_file = 'train-caltech-yolov3-tiny-config-{}.yml'.format(experiment_match.replace(caltech_yolov3_tiny,''))
            elif caltech_yolov3_tiny_infusion in experiment_match:
                config_file = 'train-caltech-yolov3-tiny-infusion-config-{}.yml'.format(experiment_match.replace(caltech_yolov3_tiny_infusion,''))
            elif caltech_yolov3_infusion in experiment_match:
                config_file = 'train-caltech-yolov3-infusion-config-{}.yml'.format(experiment_match.replace(caltech_yolov3_infusion,''))
            elif caltech_yolov3 in experiment_match and not 'tiny' in experiment_match:
                config_file = 'train-caltech-yolov3-config-{}.yml'.format(experiment_match.replace(caltech_yolov3,''))
            else:
                raise Exception("Could not find the experiment pattern.")

        else:
            print(inf_folder)
            raise Exception("Could not find the experiment number in the inference file name.")

        if not config_file:
            raise Exception('Could not understand the needed config file.')

        train_config = None
        pytrains_folder = os.path.join(main_args.framework,'grv/pytrains')
        with open(os.path.join(pytrains_folder, config_file), 'r') as stream:
            train_config = yaml.load(stream)

        if 'class_translation_path' in train_config:
            gt_file_path = train_config['test_path'].replace('.txt', '_'+train_config['class_translation_path'].replace('.yml', '.txt'))
        else:
            gt_file_path = train_config['test_path']
            # gt_file_path = 'gt_'+gt_file.replace('.txt','')
        gt_file_path = os.path.join(main_args.framework, gt_file_path)

        if not os.path.exists(gt_file_path):
            raise Exception("Missing gt file", gt_file_path)

        inference_path = os.path.join(destination_root_path, inf_folder)
        for min_overlap in main_args.min_overlap:
            command = 'python3 eval_all_inferences.py -i {} -ga {} -p {}'.format(inference_path, gt_file_path, min_overlap)
            if main_args.force_overwrite:
                command += ' -f'
            if 'class_translation_path' in train_config:
                command += ' -de ' + os.path.join(main_args.framework, train_config['test_path'])
            print('Running',command)
            p1 = subprocess.Popen(command.split(' '))
            p1.wait()
