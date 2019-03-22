import glob
import json
import os
import shutil
import operator
import sys
import argparse
import datetime
import math
import re
from collections import OrderedDict
import numpy as np

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-g', '--gt_dir', help="Converted GroundTruth dir (with convert_keras-yolo3.py)", type=str)
group.add_argument('-ga', '--gt_annot', help="Original GT annot file. It will be auto converted.", type=str)
parser.add_argument('-i', '--inference_dir', help="Annotation directory", type=str, required=True)
parser.add_argument('-p', '--min_overlap', help="Minimum overlap", type=float, default=0.5)
parser.add_argument('-r', '--global_results_file', help="Global results file", type=str, default='global_results.txt')
parser.add_argument('-f', '--force_overwrite', help="Overwrite already generated resources.", action="store_true")
parser.add_argument('-de', '--detranslate_from', help="Keras dataset file with all original and non-translated classes. This will save additional data about the original class for GT bboxes.", type=str, required=False, default=False)
parser.add_argument("-ca", "--canonical_bboxes", required=False, action="store_true", help="The training configuration.")
parser.add_argument('--img_width', help="Image Width", type=int, default=None)
parser.add_argument('--img_height', help="Image Height", type=int, default=None)
main_args = parser.parse_args()

if main_args.canonical_bboxes and not (main_args.img_width and main_args.img_height):
    raise Exception('To use canonical bboxes you need to inform the --img_width and --img_height.')

MINOVERLAP = main_args.min_overlap

with open('extra/class_list_pti01v3.txt', 'r') as class_file:
    class_map = [c.strip() for c in class_file.readlines()]

def eval_inference(no_animation=True, no_plot=True, quiet=True, ignore=None, set_class_iou=None, predicted_dir=None, groundtruth_dir=None, global_map_result_file_path=None, overwrite=False, output_version=None, first_inference=False, detranslation_dict=None):
    # if there are no classes to ignore then replace None by empty list

    if ignore is None:
      ignore = []

    specific_iou_flagged = False
    if set_class_iou is not None:
      specific_iou_flagged = True

    # if there are no images then no animation can be shown
    img_path = 'images'
    if os.path.exists(img_path):
      for dirpath, dirnames, files in os.walk(img_path):
        if not files:
          # no image files found
          no_animation = True
    else:
      no_animation = True

    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = False
    if not no_animation:
      try:
        import cv2
        show_animation = True
      except ImportError:
        print("\"opencv-python\" not found, please install to visualize the results.")
        no_animation = True

    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = False
    if not no_plot:
      try:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        draw_plot = True
      except ImportError:
        print("\"matplotlib\" not found, please install it to get the resulting plots.")
        no_plot = True

    def log_average_miss_rate(precision, fp_cumsum, num_images):
        """
            log-average miss rate:
                Calculated by averaging miss rates at 9 evenly spaced FPPI points
                between 10e-2 and 10e0, in log-space.

            output:
                    lamr | log-average miss rate
                    mr | miss rate
                    fppi | false positives per image

            references:
                [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
                   State of the Art." Pattern Analysis and Machine Intelligence, IEEE
                   Transactions on 34.4 (2012): 743 - 761.
        """

        # if there were no detections of that class
        if precision.size == 0:
            lamr = 0
            mr = 1
            fppi = 0
            return lamr, mr, fppi

        print('LEN',len(fp_cumsum), num_images, len(precision))

        fppi = fp_cumsum / float(num_images)
        mr = (1 - precision)
        print("fppi,mr",len(fppi),len(mr))
        print("fppi,mr",fppi[1],mr[-1],mr[-2])

        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)

        # Use 9 evenly spaced reference points in log-space
        ref = np.logspace(-2.0, 0.0, num = 9)
        for i, ref_i in enumerate(ref):
            # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]

        # log(0) is undefined, so we use the np.maximum(1e-10, ref)
        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

        return lamr, mr, fppi

    """
     throw error and exit
    """
    def error(msg):
      print(msg)
      sys.exit(0)

    """
     check if the number is a float between 0.0 and 1.0
    """
    def is_float_between_0_and_1(value):
      try:
        val = float(value)
        if val > 0.0 and val < 1.0:
          return True
        else:
          return False
      except ValueError:
        return False

    """
     Calculate the AP given the recall and precision array
      1st) We compute a version of the measured precision/recall curve with
           precision monotonically decreasing
      2nd) We compute the AP as the area under this curve by numerical integration.
    """
    def voc_ap(rec, prec):
      """
      --- Official matlab code VOC2012---
      mrec=[0 ; rec ; 1];
      mpre=[0 ; prec ; 0];
      for i=numel(mpre)-1:-1:1
          mpre(i)=max(mpre(i),mpre(i+1));
      end
      i=find(mrec(2:end)~=mrec(1:end-1))+1;
      ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
      """
      rec.insert(0, 0.0) # insert 0.0 at begining of list
      rec.append(1.0) # insert 1.0 at end of list
      mrec = rec[:]
      prec.insert(0, 0.0) # insert 0.0 at begining of list
      prec.append(0.0) # insert 0.0 at end of list
      mpre = prec[:]
      """
       This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab:  for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
      """
      # matlab indexes start in 1 but python in 0, so I have to do:
      #   range(start=(len(mpre) - 2), end=0, step=-1)
      # also the python function range excludes the end, resulting in:
      #   range(start=(len(mpre) - 2), end=-1, step=-1)
      for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
      """
       This part creates a list of indexes where the recall changes
        matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
      """
      i_list = []
      for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
          i_list.append(i) # if it was matlab would be i + 1
      """
       The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
      """
      ap = 0.0
      for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
      return ap, mrec, mpre


    """
     Convert the lines of a file to a list
    """
    def file_lines_to_list(path):
      # open txt file lines to a list
      with open(path) as f:
        content = f.readlines()
      # remove whitespace characters like `\n` at the end of each line
      content = [x.strip() for x in content]
      return content

    """
     Draws text in image
    """
    def draw_text_in_image(img, text, pos, color, line_width):
      font = cv2.FONT_HERSHEY_PLAIN
      fontScale = 1
      lineType = 1
      bottomLeftCornerOfText = pos
      cv2.putText(img, text,
          bottomLeftCornerOfText,
          font,
          fontScale,
          color,
          lineType)
      text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
      return img, (line_width + text_width)

    """
     Plot - adjust axes
    """
    def adjust_axes(r, t, fig, axes):
      # get text width for re-scaling
      bb = t.get_window_extent(renderer=r)
      text_width_inches = bb.width / fig.dpi
      # get axis width in inches
      current_fig_width = fig.get_figwidth()
      new_fig_width = current_fig_width + text_width_inches
      propotion = new_fig_width / current_fig_width
      # get axis limit
      x_lim = axes.get_xlim()
      axes.set_xlim([x_lim[0], x_lim[1]*propotion])

    """
     Draw plot using Matplotlib
    """
    def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
      # sort the dictionary by decreasing value, into a list of tuples
      sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
      # unpacking the list of tuples into two lists
      sorted_keys, sorted_values = zip(*sorted_dic_by_value)
      #
      if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
          fp_sorted.append(dictionary[key] - true_p_bar[key])
          tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
          fp_val = fp_sorted[i]
          tp_val = tp_sorted[i]
          fp_str_val = " " + str(fp_val)
          tp_str_val = fp_str_val + " " + str(tp_val)
          # trick to paint multicolor with offset:
          #   first paint everything and then repaint the first number
          t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
          plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
          if i == (len(sorted_values)-1): # largest bar
            adjust_axes(r, t, fig, axes)
      else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
          str_val = " " + str(val) # add a space before
          if val < 1.0:
            str_val = " {0:.2f}".format(val)
          t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
          # re-set axes to show number inside the figure
          if i == (len(sorted_values)-1): # largest bar
            adjust_axes(r, t, fig, axes)
      # set window title
      fig.canvas.set_window_title(window_title)
      # write classes in y axis
      tick_font_size = 12
      plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
      """
       Re-scale height accordingly
      """
      init_height = fig.get_figheight()
      # comput the matrix height in points and inches
      dpi = fig.dpi
      height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
      height_in = height_pt / dpi
      # compute the required figure height
      top_margin = 0.15    # in percentage of the figure height
      bottom_margin = 0.05 # in percentage of the figure height
      figure_height = height_in / (1 - top_margin - bottom_margin)
      # set new height
      if figure_height > init_height:
        fig.set_figheight(figure_height)

      # set plot title
      plt.title(plot_title, fontsize=14)
      # set axis titles
      # plt.xlabel('classes')
      plt.xlabel(x_label, fontsize='large')
      # adjust size of window
      fig.tight_layout()
      # save the plot
      fig.savefig(output_path)
      # show image
      if to_show:
        plt.show()
      # close the plot
      plt.close()

    """
     Create a "tmp_files/" and "results/" directory
    """
    tmp_files_path = "/tmp/tmp_files_{}".format(output_version)
    if not os.path.exists(tmp_files_path): # if it doesn't exist already
      os.makedirs(tmp_files_path)
    results_files_path = os.path.join(main_args.inference_dir, os.path.join('results_min-iou-{}'.format(MINOVERLAP),os.path.basename(predicted_dir)))
    # print('results_files_path',results_files_path)
    if overwrite and os.path.exists(results_files_path):
          shutil.rmtree(results_files_path)
    elif os.path.exists(results_files_path):
        print('Skipping generating results.')
        return

    # os.makedirs(results_files_path)
    if draw_plot:
      os.makedirs(results_files_path + "/classes")
    if show_animation:
      os.makedirs(results_files_path + "/images")


    hist_gt_areas = [] #areas of each gt bbox.
    hist_tp_gt_areas = [] #areas of each gt bbox marked as tp. This are GT bboxes.
    hist_tp_pred_areas = [] #areas of each pred bbox marked as tp. This are Prediction bboxes.

    hist_gt_height = [] #height of each gt bbox
    hist_tp_gt_height = [] #height of each gt bbox marked as tp. This are GT bboxes.
    hist_tp_pred_height = [] #height of each pred bbox marked as tp. This are Prediction bboxes.

    hist_gt_ratio = [] #the h/w ratio of each gt bbox.
    hist_tp_gt_ratio = [] #the h/w ratio of each gt bbox marked as tp. This are GT bboxes.

    """
    """
    results_mapping_data = []
    GT_FN_COUNTER = 0

    def detranslate_gt_class(img_path, obj_bbox):
        img_path += '.jpg'
        img_path = img_path.replace('__', '/')
        bboxes = detranslation_dict[img_path]
        for b in bboxes:
            #obj_bbox 474.0,28.0,505.0,101.0
            if obj_bbox == '{},{},{},{}'.format(b[0],b[1],b[2],b[3]):
                return b[4] #class

        raise Exception('Could not find a bbox match when detranslating')

    def save_result(img_path, obj_bbox, result, gt_obj_bbox=None, class_name=None):
        obj_bbox = obj_bbox.replace(' ', ',')
        gt_obj_bbox = gt_obj_bbox.replace(' ', ',') if gt_obj_bbox else None

        detranslated_class = None
        if detranslation_dict and 'GT' in result:
            detranslated_class = detranslate_gt_class(img_path, obj_bbox)
        elif detranslation_dict and ('TP' == result or 'FP-MULT' == result):
            detranslated_class = detranslate_gt_class(img_path, gt_obj_bbox)

        class_id = class_map.index(class_name) if class_name else None
        results_mapping_data.append('{} {} {} {} {} {}'.format(img_path, obj_bbox, result, gt_obj_bbox, class_id, detranslated_class))

    def persist_result(class_name):
        # print('GT_FN_COUNTER',GT_FN_COUNTER)
        results_mapping_path = os.path.join(results_files_path,'results_mapping_{}.txt'.format(class_name))
        with open(results_mapping_path, 'w') as res_f:
            for data in results_mapping_data:
                res_f.write(data + '\n')




    """
    Calculate area of a given bbox
    """
    def bbox_area(obj_bbox):
        bbox = [ float(x) for x in obj_bbox.split() ]
        return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)

    """
    Calculate the height of a given bbox
    """
    def bbox_height(obj_bbox):
        bbox = [ float(x) for x in obj_bbox.split() ]
        return (bbox[3] - bbox[1] + 1)

    """
    Calculate the bbox ratio (h/w)
    """
    def bbox_ratio(obj_bbox):
        #bbox -> x_min, y_min, x_max, y_max
        bbox = [ float(x) for x in obj_bbox.split() ]
        #(y_max - y_min)/(x_max - x_min)
        # print(obj_bbox)
        return (bbox[3] - bbox[1]+1)/(bbox[2] - bbox[0]+1) #+1 to avoid zero divisions

    """
     Ground-Truth
       Load each of the ground-truth files into a temporary ".json" file.
       Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    # print(os.path.join(groundtruth_dir,'*.txt'))
    ground_truth_files_list = glob.glob(os.path.join(groundtruth_dir,'*.txt'))
    print('ground_truth_files_list', len(ground_truth_files_list))

    if not  os.path.exists(groundtruth_dir):
        error('Error: Seems like you did not create the GT files yet: {}'.format(groundtruth_dir))

    if len(ground_truth_files_list) == 0:
      error("Error: No ground-truth files found! Do we have {}?".format(groundtruth_dir))
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    tmp_counter = 0
    wanted_counter = 113
    wanted_fileid = '__home__grvaliati__workspace__datasets__caltech__parsers__caltech-pedestrian-dataset-to-yolo-format-converter__images__set06_V001_749'
    for txt_file in ground_truth_files_list:
      tmp_counter += 1
      # print(txt_file)
      # print(tmp_counter)
      file_id = txt_file.split(".txt",1)[0]
      if tmp_counter == wanted_counter:
          print('>>>>>>>>>>>>',file_id)
      file_id = os.path.basename(os.path.normpath(file_id))
      # check if there is a correspondent predicted objects file
      if not os.path.exists(os.path.join(predicted_dir,file_id + ".txt")):
        error_msg = "Error. File not found: " + os.path.join(predicted_dir,file_id + ".txt") +"\n"
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
        error(error_msg)
      lines_list = file_lines_to_list(txt_file)
      # create ground-truth dictionary
      bounding_boxes = []
      is_difficult = False
      already_seen_classes = []
      for line in lines_list:
        try:
          if "difficult" in line:
              class_name, left, top, right, bottom, _difficult = line.split()
              is_difficult = True
          else:
              class_name, left, top, right, bottom = line.split()
        except ValueError:
          error_msg = "Error: File " + txt_file + " in the wrong format.\n"
          error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
          error_msg += " Received: " + line
          error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
          error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
          error(error_msg)
        # check if class is in the ignore list, if yes skip
        if class_name in ignore:
          continue
        bbox = left + " " + top + " " + right + " " +bottom
        if is_difficult:
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
            is_difficult = False
        else:
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
            hist_gt_areas.append(bbox_area(bbox))
            hist_gt_height.append(bbox_height(bbox))
            hist_gt_ratio.append(bbox_ratio(bbox))
            save_result(file_id, bbox, 'GT', class_name=class_name)
            # count that object
            if class_name in gt_counter_per_class:
              gt_counter_per_class[class_name] += 1
            else:
              # if class didn't exist yet
              gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)


      # dump bounding_boxes into a ".json" file
      with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

      if tmp_counter == wanted_counter:
          print('>>>>>>>>>>>>bounding_boxes', len(bounding_boxes), bounding_boxes)

    gt_classes = list(gt_counter_per_class.keys())
    gt_total = sum(gt_counter_per_class.values())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    #print(gt_classes)
    print('gt_counter_per_class',gt_counter_per_class)

    """
     Check format of the flag --set-class-iou (if used)
      e.g. check if class exists
    """
    if specific_iou_flagged:
      n_args = len(set_class_iou)
      error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
      if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
      # [class_1] [IoU_1] [class_2] [IoU_2]
      # specific_iou_classes = ['class_1', 'class_2']
      specific_iou_classes = set_class_iou[::2] # even
      # iou_list = ['IoU_1', 'IoU_2']
      iou_list = set_class_iou[1::2] # odd
      if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
      for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
              error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
      for num in iou_list:
        if not is_float_between_0_and_1(num):
          error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    """
     Predicted
       Load each of the predicted files into a temporary ".json" file.
    """
    # get a list with the predicted files
    predicted_files_list = glob.glob(os.path.join(predicted_dir, '*.txt'))
    predicted_files_list.sort()

    print('predicted_files_list', len(predicted_files_list))

    for class_index, class_name in enumerate(gt_classes):
      bounding_boxes = []
      for txt_file in predicted_files_list:
        #print(txt_file)
        # the first time it checks if all the corresponding ground-truth files exist
        file_id = txt_file.split(".txt",1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        if class_index == 0:
          if not os.path.exists(os.path.join(groundtruth_dir, file_id + ".txt")):
            error_msg = "Error. File not found: "+os.path.join(groundtruth_dir, file_id + ".txt")+"\n"
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
            error(error_msg)
        lines = file_lines_to_list(txt_file)
        for line in lines:
          try:
            tmp_class_name, confidence, left, top, right, bottom = line.split()
          except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format.\n"
            error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
            error_msg += " Received: " + line
            error(error_msg)
          if tmp_class_name == class_name:
            #print("match")
            bbox = left + " " + top + " " + right + " " +bottom
            bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
            #print(bounding_boxes)
      # sort predictions by decreasing confidence
      bounding_boxes.sort(key=lambda x:x['confidence'], reverse=True)
      with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    wap_dictionary = {}
    lamr_dictionary = {}

    # open file to store the results
    with open(results_files_path + "/results.txt", 'w') as results_file:
      results_file.write("# AP and precision/recall per class\n")
      count_true_positives = {}

      for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
         Load predictions of that class
        """
        predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
        predictions_data = json.load(open(predictions_file))

        """
         Assign predictions to ground truth objects
        """
        nd = len(predictions_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, prediction in enumerate(predictions_data):
          file_id = prediction["file_id"]
          if file_id == wanted_fileid:
              print('wanted idx', idx)
          if show_animation:
            # find ground truth image
            ground_truth_img = glob.glob1(img_path, file_id + ".*")
            #tifCounter = len(glob.glob1(myPath,"*.tif"))
            if len(ground_truth_img) == 0:
              error("Error. Image not found with id: " + file_id)
            elif len(ground_truth_img) > 1:
              error("Error. Multiple image with id: " + file_id)
            else: # found image
              #print(img_path + "/" + ground_truth_img[0])
              # Load image
              img = cv2.imread(img_path + "/" + ground_truth_img[0])
              # Add bottom border to image
              bottom_border = 60
              BLACK = [0, 0, 0]
              img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
          # assign prediction to ground truth object if any
          #   open ground-truth with that file_id
          gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
          ground_truth_data = json.load(open(gt_file))
          ovmax = -1
          gt_match = -1
          # load prediction bounding-box
          bb = [ float(x) for x in prediction["bbox"].split() ]
          for obj in ground_truth_data:
            # look for a class_name match
            if obj["class_name"] == class_name:
              bbgt = [ float(x) for x in obj["bbox"].split() ]
              bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
              iw = bi[2] - bi[0] + 1
              ih = bi[3] - bi[1] + 1
              if iw > 0 and ih > 0:
                # compute overlap (IoU) = area of intersection / area of union
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                ov = iw * ih / ua
                if ov > ovmax:
                  ovmax = ov
                  gt_match = obj

          # assign prediction as true positive/don't care/false positive
          if show_animation:
            status = "NO MATCH FOUND!" # status is only used in the animation
          # set minimum overlap
          min_overlap = MINOVERLAP
          if specific_iou_flagged:
            if class_name in specific_iou_classes:
              index = specific_iou_classes.index(class_name)
              min_overlap = float(iou_list[index])
          if ovmax >= min_overlap:
            if "difficult" not in gt_match:
                if not bool(gt_match["used"]):
                  # true positive
                  tp[idx] = 1
                  gt_match["used"] = True
                  count_true_positives[class_name] += 1
                  hist_tp_gt_areas.append(bbox_area(gt_match["bbox"]))
                  hist_tp_gt_height.append(bbox_height(gt_match["bbox"]))
                  hist_tp_pred_areas.append(bbox_area(prediction["bbox"]))
                  hist_tp_pred_height.append(bbox_height(prediction["bbox"]))
                  hist_tp_gt_ratio.append(bbox_ratio(gt_match["bbox"]))

                  save_result(prediction["file_id"], prediction["bbox"], 'TP', gt_match["bbox"], class_name=class_name)

                  # update the ".json" file
                  with open(gt_file, 'w') as f:
                      f.write(json.dumps(ground_truth_data))
                  if show_animation:
                    status = "MATCH!"
                else:
                  # false positive (multiple detection)
                  save_result(prediction["file_id"], prediction["bbox"], 'FP-MULT', gt_match["bbox"], class_name=class_name)
                  fp[idx] = 1
                  if show_animation:
                    status = "REPEATED MATCH!"

          else:
            # false positive
            save_result(prediction["file_id"], prediction["bbox"], 'FP-OVLP', class_name=class_name)
            fp[idx] = 1
            if ovmax > 0:
              status = "INSUFFICIENT OVERLAP"

          if prediction['file_id'] == wanted_fileid:
            print('prediction', prediction)
            print('tp',tp[idx])
            print('fp',fp[idx])
            print('rec',float(tp[idx]) / gt_counter_per_class[class_name])
            print('prec',float(tp[idx]) / (fp[idx] + tp[idx]))
            print('gt_counter_per_class[class_name]',gt_counter_per_class[class_name])


          """
           Draw image to show animation
          """
          if show_animation:
            height, widht = img.shape[:2]
            # colors (OpenCV works with BGR)
            white = (255,255,255)
            light_blue = (255,200,100)
            green = (0,255,0)
            light_red = (30,30,255)
            # 1st line
            margin = 10
            v_pos = int(height - margin - (bottom_border / 2))
            text = "Image: " + ground_truth_img[0] + " "
            img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
            text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
            img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
            if ovmax != -1:
              color = light_red
              if status == "INSUFFICIENT OVERLAP":
                text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
              else:
                text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                color = green
              img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
            # 2nd line
            v_pos += int(bottom_border / 2)
            rank_pos = str(idx+1) # rank position (idx starts at 0)
            text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(prediction["confidence"])*100)
            img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
            color = light_red
            if status == "MATCH!":
              color = green
            text = "Result: " + status + " "
            img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

            if ovmax > 0: # if there is intersections between the bounding-boxes
              bbgt = [ float(x) for x in gt_match["bbox"].split() ]
              cv2.rectangle(img,(int(bbgt[0]),int(bbgt[1])),(int(bbgt[2]),int(bbgt[3])),light_blue,2)
            if status == "MATCH!":
              cv2.rectangle(img,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),green,2)
            else:
              cv2.rectangle(img,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),light_red,2)
            cv2.imshow("Animation", img)
            cv2.waitKey(20) # show image for 20 ms
            # save image to results
            output_img_path = results_files_path + "/images/" + class_name + "_prediction" + str(idx) + ".jpg"
            cv2.imwrite(output_img_path, img)




        #selected missed GT
        for txt_file in ground_truth_files_list:
            #print(txt_file)
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            if file_id == wanted_fileid:
                print(ground_truth_data)

            unused_gt = [obj for obj in ground_truth_data if not obj['used']]
            # print('unused_gt',len(unused_gt))
            for obj in unused_gt:
                if class_name != obj["class_name"]:
                    continue
                GT_FN_COUNTER += 1
                save_result(file_id, obj["bbox"], 'GT-FN', class_name=obj["class_name"])

        persist_result(class_name)
        results_mapping_data = []

        print('len tp fp', len(tp), len(fp))

        for i in range(10):
            print('fp/tp')
            print(fp[i], tp[i])

        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
          fp[idx] += cumsum
          cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
          tp[idx] += cumsum
          cumsum += val
        #print(tp)
        for i in range(10):
            print('fp/tp cumsum')
            print(fp[i], tp[i])

        print('gt_counter_per_class[class_name]',gt_counter_per_class[class_name])

        rec = tp[:]
        for idx, val in enumerate(tp):
          rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
          prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        for i in range(10):
            print('prec(fp)/rec(tp)')
            print(prec[i], rec[i])

        print('BEFORE VOCAP', len(rec), len(prec))

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

        for i in range(10):
            print('mprec/mrec')
            print(mprec[i], mrec[i],)

        # print(text)
        #write global result
        # with open(global_map_result_file_path, 'a') as global_f:
        #     global_f.write(text + "\n")
        """
         Write to results.txt
        """
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
        if not quiet:
          print(text)

        # print('gt_counter_per_class',gt_counter_per_class)
        ap_dictionary[class_name] = ap
        wap_dictionary[class_name] = ap * gt_counter_per_class[class_name]
        # print('gt_total', gt_total)
        # print('ap_dictionary', ap_dictionary)
        # print('wap_dictionary', wap_dictionary)

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
        lamr_dictionary[class_name] = lamr

        npys_output_folder = '/media/gustavo/GRV/workspace/mestrado/inferences/npys'
        np.save('{}/{}_{}_{}_mr.npy'.format(npys_output_folder, os.path.basename(predicted_dir), class_name, main_args.min_overlap), mr)
        np.save('{}/{}_{}_{}_fppi.npy'.format(npys_output_folder, os.path.basename(predicted_dir), class_name, main_args.min_overlap), fppi)
        np.save('{}/{}_{}_{}_fp.npy'.format(npys_output_folder, os.path.basename(predicted_dir), class_name, main_args.min_overlap), fp)
        np.save('{}/{}_{}_{}_tp.npy'.format(npys_output_folder, os.path.basename(predicted_dir), class_name, main_args.min_overlap), tp)


        print('lamr_dictionary', lamr_dictionary)

        for i in range(10):
            print('mr, fppi')
            print(mr[i], fppi[i])

        """
         Draw plot
        """
        if draw_plot:
          plt.plot(rec, prec, '-o')
          # add a new penultimate point to the list (mrec[-2], 0.0)
          # since the last line segment (and respective area) do not affect the AP value
          area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
          area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
          plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
          # set window title
          fig = plt.gcf() # gcf - get current figure
          fig.canvas.set_window_title('AP ' + class_name)
          # set plot title
          plt.title('class: ' + text)
          #plt.suptitle('This is a somewhat long figure title', fontsize=16)
          # set axis titles
          plt.xlabel('Recall')
          plt.ylabel('Precision')
          # optional - set axes
          axes = plt.gca() # gca - get current axes
          axes.set_xlim([0.0,1.0])
          axes.set_ylim([0.0,1.05]) # .05 to give some extra space
          # Alternative option -> wait for button to be pressed
          #while not plt.waitforbuttonpress(): pass # wait for key display
          # Alternative option -> normal display
          #plt.show()
          # save the plot
          fig.savefig(results_files_path + "/classes/" + class_name + ".png")
          plt.cla() # clear axes for next plot

          """
          ROC Curve
          """

          # print(len(fp),len(tp))
          plt.plot([0, 1], [0, 1], 'k--')
          print(len(fp),len(tp))
          plt.plot(fp, tp)
          plt.xlabel('False positive rate')
          plt.ylabel('True positive rate')
          plt.title('ROC curve')
          plt.legend(loc='best')
          fig.savefig(results_files_path + "/test_grv_" + class_name + ".png")
          plt.cla() # clear axes for next plot

          """
          Miss Rate vs FPPI curve
          """
          # n_fppi = np.insert(fppi, -1, 0)
          # n_fppi = np.insert(n_fppi, -1, 0)
          # print(len(fp),len(tp))
          # plt.plot([0, 1], [0, 1], 'k--')
          plt.plot(fppi, mr)
          # plt.plot(fppi, ys_inverse)
          plt.gca().set_xscale('log')
          plt.grid()

          plt.xlabel('False Positives Per Image')
          plt.ylabel('Miss Rate')
          plt.title('Miss Rate x FPPI curve')
          plt.legend(loc='best')
          fig.savefig(results_files_path + "/missrate-fppi_grv_" + class_name + ".png")
          plt.cla() # clear axes for next plot



      if show_animation:
        cv2.destroyAllWindows()

      results_file.write("\n# mAP of all classes\n")
      mAP = sum_AP / n_classes
      text = "mAP = {0:.2f}%".format(mAP*100)
      results_file.write(text + "\n")
      print(text)

      results_file.write("\n# wAP of all classes\n")
      wAP = sum(wap_dictionary.values()) / gt_total
      text = "wAP = {0:.2f}%".format(wAP*100)
      results_file.write(text + "\n")
      print(text)

      results_file.write("\n# aAP of all classes\n")
      # print('TP bboxes', sum(count_true_positives.values()))
      aAP = sum(count_true_positives.values()) / gt_total
      text = "aAP = {0:.2f}%".format(aAP*100)
      results_file.write(text + "\n")
      print(text)

      epoch_regex = re.compile('(ep[0-9]{3}|weights_final)')
      epoch = epoch_regex.search(os.path.basename(predicted_dir)).group()
      if not epoch:
          raise Exception('Could not find the epoch number in the inference file name.')

      #write global result
      with open(global_map_result_file_path, 'a') as global_f:
          ordered_ap_dictionary = OrderedDict(sorted(ap_dictionary.items()))
          ordered_lamr_dictionary = OrderedDict(sorted(lamr_dictionary.items()))
          if first_inference:
              #write headers.
              global_f.write("inference_dir;gt_dir;min_iou;epoch;mAP;wAP;aAP")

              for cl in ordered_ap_dictionary:
                  global_f.write(";{}".format(cl))

              for cl in ordered_lamr_dictionary:
                  global_f.write(";lamr-{}".format(cl))

              global_f.write("\n")

          #write scores.
          global_f.write("{};{};{};{};{:.4f};{:.4f};{:.4f}".format(
            os.path.basename(main_args.inference_dir),
            os.path.basename(main_args.gt_dir if main_args.gt_dir else main_args.gt_annot),
            main_args.min_overlap,
            epoch,
            mAP,
            wAP,
            aAP))
          for cl in ordered_ap_dictionary:
              global_f.write(";{:.4f}".format(ordered_ap_dictionary[cl]))

          for cl in ordered_lamr_dictionary:
              global_f.write(";{:.4f}".format(ordered_lamr_dictionary[cl]))

          global_f.write("\n")

    shutil.rmtree(tmp_files_path)

    """
     Count total of Predictions
    """
    # iterate through all the files
    pred_counter_per_class = {}
    #all_classes_predicted_files = set([])
    for txt_file in predicted_files_list:
      # get lines to list
      lines_list = file_lines_to_list(txt_file)
      for line in lines_list:
        class_name = line.split()[0]
        # check if class is in the ignore list, if yes skip
        if class_name in ignore:
          continue
        # count that object
        if class_name in pred_counter_per_class:
          pred_counter_per_class[class_name] += 1
        else:
          # if class didn't exist yet
          pred_counter_per_class[class_name] = 1
    print('pred_counter_per_class', pred_counter_per_class)
    pred_classes = list(pred_counter_per_class.keys())


    """
     Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
      window_title = "Ground-Truth Info"
      plot_title = "Ground-Truth\n"
      plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
      x_label = "Number of objects per class"
      output_path = results_files_path + "/Ground-Truth Info.png"
      to_show = False
      plot_color = 'forestgreen'
      draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
        )

    """
     Write number of ground-truth objects per class to results.txt
    """
    with open(results_files_path + "/results.txt", 'a') as results_file:
      results_file.write("\n# Number of ground-truth objects per class\n")
      for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
     Finish counting true positives
    """
    for class_name in pred_classes:
      # if class exists in predictions but not in ground-truth then there are no true positives in that class
      if class_name not in gt_classes:
        count_true_positives[class_name] = 0
    #print(count_true_positives)

    """
     Plot the total number of occurences of each class in the "predicted" folder
    """
    if draw_plot:
      window_title = "Predicted Objects Info"
      # Plot title
      plot_title = "Predicted Objects\n"
      plot_title += "(" + str(len(predicted_files_list)) + " files and "
      count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
      plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
      # end Plot title
      x_label = "Number of objects per class"
      output_path = results_files_path + "/Predicted Objects Info.png"
      to_show = False
      plot_color = 'forestgreen'
      true_p_bar = count_true_positives
      draw_plot_func(
        pred_counter_per_class,
        len(pred_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
        )

    """
     Write number of predicted objects per class to results.txt
    """
    with open(results_files_path + "/results.txt", 'a') as results_file:
      results_file.write("\n# Number of predicted objects per class\n")
      for class_name in sorted(pred_classes):
        n_pred = pred_counter_per_class[class_name]
        text = class_name + ": " + str(n_pred)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
        results_file.write(text)

    """
     Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = results_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    """
     Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
      window_title = "mAP"
      plot_title = "mAP = {0:.2f}%".format(mAP*100)
      x_label = "Average Precision"
      output_path = results_files_path + "/mAP.png"
      to_show = True
      plot_color = 'royalblue'
      draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )

    """
    Plot the histograms of bbox areas by GT and TP
    """
    draw_plot = False
    if draw_plot:
        """
        Histogram: GT x TP
        This will reveal in which bbox sizes (areas) it is more problematic to detect.
        """
        plt.title('GT x TP bbox areas. BINS=auto')
        plt.xlim(0, 60000)
        _, bins, _ = plt.hist(hist_gt_areas, bins='auto', label='GT {}'.format(len(hist_gt_areas)))
        plt.hist(hist_tp_gt_areas, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(hist_tp_gt_areas)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=4)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox area')
        plt.savefig(os.path.join(results_files_path,'hist_area_gt_tp_binsauto.jpg'),dpi = 300)
        plt.cla()

        n_bins = 50
        plt.title('GT x TP bbox areas. BINS={}'.format(n_bins))
        plt.xlim(0, 60000)
        _, bins, _ = plt.hist(hist_gt_areas, bins=n_bins, label='GT {}'.format(len(hist_gt_areas)))
        plt.hist(hist_tp_gt_areas, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(hist_tp_gt_areas)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox area')
        plt.savefig(os.path.join(results_files_path,'hist_area_gt_tp_bins-manual.jpg'),dpi = 600)
        plt.cla()

        less2k_hist_gt_areas = [i for i in hist_gt_areas if i <= 2000]
        less2k_hist_tp_gt_areas = [i for i in hist_tp_gt_areas if i <= 2000]
        n_bins = 50
        plt.title('GT x TP bbox areas. BINS={}. Zoom <= 2k.'.format(n_bins))
        _, bins, _ = plt.hist(less2k_hist_gt_areas, bins=n_bins, label='GT {}'.format(len(less2k_hist_gt_areas)))
        plt.hist(less2k_hist_tp_gt_areas, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(less2k_hist_tp_gt_areas)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox area')
        plt.savefig(os.path.join(results_files_path,'hist_area_gt_tp_zoom_bins-manual.jpg'),dpi = 600)
        plt.cla()

        n_bins = 50
        plt.title('GT x TP-Predictions bbox areas. BINS={}'.format(n_bins))
        plt.xlim(0, 60000)
        _, bins, _ = plt.hist(hist_gt_areas, bins=n_bins, label='GT {}'.format(len(hist_gt_areas)))
        plt.hist(hist_tp_pred_areas, facecolor='pink', bins=bins, alpha=0.5, label='TP {}'.format(len(hist_tp_pred_areas)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox area')
        plt.savefig(os.path.join(results_files_path,'hist_area_gt_tp-pred_bins-manual.jpg'),dpi = 600)
        plt.cla()



        """
        Histogram: GT HEIGHTS x TP HEIGHTS
        This will reveal in which PEDESTRIAN HEIGHTS it is more problematic to detect.
        """

        plt.title('GT x TP bbox heights. BINS=auto')
        _, bins, _ = plt.hist(hist_gt_height, bins='auto', label='GT {}'.format(len(hist_gt_height)))
        plt.hist(hist_tp_gt_height, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(hist_tp_gt_height)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox height')
        plt.savefig(os.path.join(results_files_path,'hist_height_gt_tp_bins-auto.jpg'),dpi = 300)
        plt.cla()

        n_bins = 50
        plt.title('GT x TP bbox heights. BINS={}'.format(n_bins))
        _, bins, _ = plt.hist(hist_gt_height, bins=n_bins, label='GT {}'.format(len(hist_gt_height)))
        plt.hist(hist_tp_gt_height, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(hist_tp_gt_height)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox height')
        plt.savefig(os.path.join(results_files_path,'hist_height_gt_tp_bins-manual.jpg'),dpi = 600)
        plt.cla()

        less2k_hist_gt_height = [i for i in hist_gt_height if i <= 2000]
        less2k_hist_tp_gt_height = [i for i in hist_tp_gt_height if i <= 2000]
        n_bins = 50
        plt.title('GT x TP bbox heights. Zoom <= 2k. BINS={}'.format(n_bins))
        _, bins, _ = plt.hist(less2k_hist_gt_height, bins=n_bins, label='GT {}'.format(len(less2k_hist_gt_height)))
        plt.hist(less2k_hist_tp_gt_height, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(less2k_hist_tp_gt_height)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox height')
        plt.savefig(os.path.join(results_files_path,'hist_height_gt_tp_zoom_bins-manual.jpg'),dpi = 600)
        plt.cla()

        x_limit = 300
        less_x_hist_gt_height = [i for i in hist_gt_height if i <= x_limit]
        less_x_hist_tp_gt_height = [i for i in hist_tp_gt_height if i <= x_limit]
        plt.title('GT x TP bbox heights. Step 25. Zoom <= {}.'.format(x_limit))
        _, bins, _ = plt.hist(less_x_hist_gt_height, bins=np.arange(0,x_limit,25), label='GT {}'.format(len(less_x_hist_gt_height)))
        plt.hist(less_x_hist_tp_gt_height, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(less_x_hist_tp_gt_height)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox height')
        plt.savefig(os.path.join(results_files_path,'hist_height_gt_tp_zoom_step-25.jpg'),dpi = 600)
        plt.cla()

        n_bins = 50
        plt.title('GT x TP-Predictions bbox heights. BINS={}'.format(n_bins))
        _, bins, _ = plt.hist(hist_gt_height, bins=n_bins, label='GT {}'.format(len(hist_gt_height)))
        plt.hist(hist_tp_pred_height, facecolor='pink', bins=bins, alpha=0.5, label='TP-Pred {}'.format(len(hist_tp_pred_height)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox height')
        plt.savefig(os.path.join(results_files_path,'hist_height_gt_tp-pred_bins-manual.jpg'),dpi = 600)
        plt.cla()

        x_limit = 5
        less_hist_gt_ratio = [i for i in hist_gt_ratio if i <= x_limit]
        less_hist_tp_gt_ratio = [i for i in hist_tp_gt_ratio if i <= x_limit]
        plt.title('bbox ratios (h/w) for GT and TP: zoomed in range 0 to {}'.format(x_limit))
        _, bins, _ = plt.hist(less_hist_gt_ratio, bins=np.arange(0,x_limit,0.1), label='GT {}'.format(len(less_hist_gt_ratio)))
        plt.hist(less_hist_tp_gt_ratio, facecolor='green', bins=bins, alpha=0.5, label='TP {}'.format(len(less_hist_tp_gt_ratio)))
        plt.grid(True)
        plt.xticks(bins,rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.legend(loc='upper right')
        plt.ylabel('# bboxes')
        plt.xlabel('bbox ratio')
        plt.savefig(os.path.join(results_files_path,'hist_bboxratio_gt_tp_zoom.jpg'),dpi = 600)
        plt.cla()

def get_ratio(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (y_max - y_min)/(x_max - x_min)

def normal_round(number):
    #3.5
    integer_n = int(number) #3
    float_n = number - integer_n #0.5
    if float_n >= 0.5:
        return integer_n + 1 #4
    else:
        return integer_n

def get_canonical_bboxes(original_bboxes, img_width, img_height, round_type='normal', side_ajustment='one'):
    acceptable_ratios = [1,2,3]
    canonical_bboxes = []
    for bbox in original_bboxes:
        x_min, y_min, x_max, y_max = bbox
        new_x_min, new_y_min, new_x_max, new_y_max = bbox

        if side_ajustment=='one':

            #step1: resize
            original_ratio = get_ratio(bbox)
            '''
            if the original ratio is higher than maximum acceptable_ratios, we need to increase the width.
            else we can increase the height.
            '''
            if original_ratio > max(acceptable_ratios):
                #we need to increase the width so that the height reduces to maximum acceptable_ratios.
                max_height_ratio = max(acceptable_ratios)
                original_height = y_max - y_min
                #the width should be increased to 1/max_height_ratio of the height.
                new_width = original_height // max_height_ratio
                #We need to expand it evenly in the sides.
                original_width = x_max - x_min
                width_diff = new_width - original_width #new_width is bigger.
                new_x_min = x_min - width_diff//2
                #We need to check if new_x_min still is in the image boundaries.
                if new_x_min <= 0:
                    #not enough space
                    new_x_min = 0
                    #we will expand the remaining to the other direction.
                new_x_max = new_x_min + new_width
                #lets check the same for the new_x_max
                if new_x_max >= img_width:
                    #not enough space
                    new_x_max = img_width
                    new_x_min = img_width - new_width
            else:
                #we can increase the height
                if round_type == 'normal':
                    new_height_ratio = normal_round(original_ratio) #normal rounding (up or down).
                    if new_height_ratio < 1:
                        new_height_ratio = 1
                elif round_type == 'up':
                    new_height_ratio = math.ceil(original_ratio) #rounding up
                new_height = math.ceil(new_height_ratio*(x_max - x_min)) # the ratio is relative to the width.
                #In how many pixels did the height grow?
                original_height = y_max - y_min
                height_diff = new_height - original_height
                #We need to split the growth up and down.
                #So, we put the ymin half the height_diff up.
                half_diff = height_diff // 2
                new_y_min = y_min - half_diff
                #But we check how many pixels are left upwards. We cannot overflow the img borders.
                if not (y_min - half_diff >= 0):
                    #not enough space.
                    new_y_min = 0
                #Now we have found the good new position for y_min, we add the complete needed height.
                new_y_max = new_y_min + new_height
                #We also need to check if we kept outselves the bottom image boundaries.
                if new_y_max >= img_height: #img_height does not include zero.
                    #We got out of space in the bottom. So lets move up the bbox to keep in the limits.
    #                 remaining_height = img_height - new_y_max
    #                 new_y_min -= remaining_height
    #                 new_y_max -= remaining_height
                    new_y_max = img_height
                    new_y_min = img_height - new_height

            #Lets check if we did it right, otherwise fallback to the original bbox.
            if not (new_x_min >= 0 and new_y_min >= 0 and new_x_max < img_width and new_y_max < img_height):
                # messed up, fallback!
    #             print('Could not convert the original bbox. We are going to use the original. Original: {}. Problematic: {}'.format(bbox, [new_x_min, new_y_min, new_x_max, new_y_max]))
                canonical_bboxes.append(bbox)
            else:
                canonical_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
        elif side_ajustment=='both':
            pass

    return canonical_bboxes

def convert_inference(annotation_file, type=None, overwrite=False):
    PRED_TYPE='pred'
    GT_TYPE='gt'

    if not type:
        raise Exception('Missing inference type for convertion.')
    if type not in [PRED_TYPE, GT_TYPE]:
        raise Exception('Unknown type for convertion')

    if type==PRED_TYPE:
        folder_prefix = 'pred_{}'
        if main_args.canonical_bboxes:
            folder_prefix = 'canonical_'+folder_prefix
        output_path = os.path.join(main_args.inference_dir, folder_prefix.format(os.path.basename(annotation_file).replace('.txt','')))
    else:
        if annotation_file:
            output_path = 'gt_' + os.path.basename(main_args.gt_annot).replace('.txt','')
        else:
            output_path = os.path.abspath(main_args.gt_dir)

    if os.path.exists(output_path):
        if overwrite and os.path.exists(output_path):
            shutil.rmtree(output_path)
        else:
            print('Skipping inference parsing.', type, output_path)
            return output_path

    os.makedirs(output_path)

    with open(annotation_file, 'r') as annot_f:
        for annot in annot_f:
            annot = annot.split(' ')
            img_path = annot[0].strip()
            file_name = img_path.replace('.jpg', '.txt').replace('/', '__')
            output_file_path = os.path.join(output_path, file_name)
            os.path.dirname(output_file_path)

            with open(output_file_path, 'w') as out_f:
                for bbox in annot[1:]:
                    if type==GT_TYPE:
                        # Here we are dealing with ground-truth annotations
                        # <class_name> <left> <top> <right> <bottom> [<difficult>]
                        # todo: handle difficulty
                        x_min, y_min, x_max, y_max, class_id = list(map(float, bbox.split(',')))
                        # print(class_map, class_id)
                        out_box = '{} {} {} {} {}'.format(
                            class_map[int(class_id)].strip(), x_min, y_min, x_max, y_max)
                    else:
                        # Here we are dealing with predictions annotations
                        # <class_name> <confidence> <left> <top> <right> <bottom>
                        x_min, y_min, x_max, y_max, class_id, score = list(map(float, bbox.split(',')))
                        if main_args.canonical_bboxes:
                            x_min, y_min, x_max, y_max = get_canonical_bboxes(
                                [ [x_min, y_min, x_max, y_max] ],
                                img_width=main_args.img_width,
                                img_height=main_args.img_height)[0]

                        out_box = '{} {} {} {} {} {}'.format(
                            class_map[int(class_id)].strip(), score,  x_min, y_min, x_max, y_max)

                    out_f.write(out_box + "\n")

    return output_path


if __name__ == '__main__':

    output_version = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(main_args.global_results_file, 'a') as global_f:
        global_f.write("Evaluated at: {}\n".format(output_version))
        global_f.write(os.path.basename(main_args.inference_dir)+"\n")
        global_f.write("GT: " + os.path.abspath(main_args.gt_dir) if main_args.gt_dir else os.path.abspath(main_args.gt_annot) + '\n')
        global_f.write("min-iou: {}".format(main_args.min_overlap) + '\n')
        if main_args.canonical_bboxes:
            global_f.write('Canonical bboxes: the predicted bboxes evaluated have been parsed to canonical by eval_all_inferences.py.\n')

    inferences = glob.glob(os.path.join(main_args.inference_dir, 'infer_*.txt'))
    inferences.sort()

    gt_annot_file = main_args.gt_annot if main_args.gt_annot else None
    gt_converted_dir = convert_inference(gt_annot_file , type='gt', overwrite=main_args.force_overwrite)

    detranslation_dict = None
    if main_args.detranslate_from:
        detranslation_dict = {}
        with open(main_args.detranslate_from, 'r') as annot_f:
            for line in annot_f:
                splitted = line.split(' ')
                img_path = splitted[0].strip()
                detranslation_dict[img_path] = []
                for bbox in splitted[1:]:
                    x_min, y_min, x_max, y_max, class_id = list(map(float, bbox.split(',')))
                    detranslation_dict[img_path].append([x_min, y_min, x_max, y_max, class_id])


    first_inference = True
    for inference in inferences:
        print('For: ',os.path.basename(inference))
        pred_converted_inference_dir = convert_inference(inference, type='pred', overwrite=main_args.force_overwrite)
        eval_inference(
            predicted_dir=pred_converted_inference_dir,
            groundtruth_dir=gt_converted_dir,
            global_map_result_file_path=main_args.global_results_file,
            no_plot=False,
            overwrite=main_args.force_overwrite,
            output_version=output_version,
            first_inference=first_inference,
            detranslation_dict=detranslation_dict,
            )
        first_inference = False
