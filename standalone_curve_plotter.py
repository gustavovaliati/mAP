import glob
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

npys_root = '/media/gustavo/GRV/workspace/mestrado/inferences/npys'

npys_mr = glob.glob(npys_root + '/*_mr.npy')
npys_fppi = glob.glob(npys_root + '/*_fppi.npy')

npys_fp = glob.glob(npys_root + '/*_fp.npy')
npys_tp = glob.glob(npys_root + '/*_tp.npy')

'''
classes = [
    'person-no-ocl',
    'person-normal-ocl',
    'person-severe-ocl',
    'person-head',
    'person-body-part',
    'person-top-view',
    'person-angled',
    'person-far',
]
'''

def plot_missrate_fppi(file_name=None, legend_font_size=6, pti_tiny_line_style=False):
    if not file_name:
        raise Exception('Missing file_name for plotting')

    for class_name in wanted_classes:
        for iou in wanted_ious:
            for i, exp in enumerate(wanted_experiments):

                mr_npy = '{}/{}_{}_{}_mr.npy'.format(npys_root, exp, class_name, iou)
                fppi_npy = '{}/{}_{}_{}_fppi.npy'.format(npys_root, exp, class_name, iou)

                if (mr_npy in npys_mr) and (fppi_npy in npys_fppi):

                    fppi = np.load(fppi_npy)
                    mr = np.load(mr_npy)

                    if pti_tiny_line_style:
                        if '_pti01_tiny_yolo_' in exp:
                            plt.plot(fppi, mr, label=legends[i], linestyle=':')
                        else:
                            plt.plot(fppi, mr, label=legends[i], linestyle='-')
                    else:
                        plt.plot(fppi, mr, label=legends[i], linestyle='-')

                else:
                    raise Exception('Missing npy file for ' + exp)

            # plt.ylim((0,1))
            plt.legend(loc='best', prop={'size': legend_font_size})
            plt.xlabel('False Positives Per Image')
            plt.ylabel('Miss Rate')
            # plt.title('Miss Rate x FPPI curve')
            plt.gca().set_xscale('log')
            plt.grid()
            plt.gcf().savefig(npys_root + "/plots/mr-fppi_{}_{}_{}.png".format(file_name, class_name, iou), dpi=300)
            plt.cla() # clear axes for next plot

def plot_roc(file_name=None):
    if not file_name:
        raise Exception('Missing file_name for plotting')

    for class_name in wanted_classes:
        for iou in wanted_ious:
            for i, exp in enumerate(wanted_experiments):

                fp_npy = '{}/{}_{}_{}_fp.npy'.format(npys_root, exp, class_name, iou)
                tp_npy = '{}/{}_{}_{}_tp.npy'.format(npys_root, exp, class_name, iou)

                if (tp_npy in npys_tp) and (fp_npy in npys_fp):

                    fp = np.load(fp_npy)
                    tp = np.load(tp_npy)

                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.plot(fp, tp, label=legends[i])

                else:
                    raise Exception('Missing npy file for ' + exp)

            # plt.ylim((0,1))
            plt.legend(loc='lower right')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.grid()
            # plt.title('ROC curve')
            plt.gcf().savefig(npys_root + "/plots/roc_{}_{}_{}.png".format(file_name, class_name, iou), dpi=300)
            plt.cla() # clear axes for next plot



"""
############################
Miss Rate vs FPPI curve
############################
"""

'''
PLOT:
TESTING CURVE
'''
wanted_classes = [
    'person-no-ocl',
]
wanted_ious = [
    0.5
]
wanted_experiments = [
    'pred_infer_logscaltech_yolov3-seg-045_ep002_caltech_yolo_infusion_final_experiments_1_20190112051857_iou-0.45_score-0.01'
]
legends = [
    'Caltech Test'
]

plot_missrate_fppi('testing')


'''
PLOT:
Plotting for the Table 7.13.
'''

wanted_classes = [
    'person-no-ocl',
]
wanted_ious = [
    0.4
]
wanted_experiments = [
    'pred_infer_logsdefault-021_ep007_pti01_yolo_default_pti01_v3_checking-translations_one-class_20181021140404',
    'pred_infer_logsdefault-029_ep006_pti01_yolo_dataset_by_event_20181101094129',
    'pred_infer_logsdefault-014_ep001_pti01_yolo_leave-one-out_20181006113527',
    'pred_infer_logsdefault-018_ep006_pti01_yolo_leave-one-out_flip-only_20181020181920',
    'pred_infer_logsdefault-031_ep006_pti01_yolo_default_pti01_v3_leave-5-out_one-class_20181210190848',
    'pred_infer_logsdefault-032_ep004_pti01_yolo_default_pti01_leave-3-out_one-class_discarding_20181210200230',
    'pred_infer_logsdefault-033_ep009_pti01_yolo_default_pti01_high-bias_one-class_discarding_20181211193319',

    'pred_infer_logsdefault-tiny-031_ep009_pti01_tiny_yolo_data-aug-flip_only_20181006102919',
    'pred_infer_logsdefault-tiny-052_ep009_pti01_tiny_yolo_tiny-dataset_by_event-oneclass_20181210203035',
    'pred_infer_logsdefault-tiny-033_ep003_pti01_tiny_yolo_leave-one-out_20181006103049',
    'pred_infer_logsdefault-tiny-035_ep007_pti01_tiny_yolo_leave-one-out_20181006114605',
    'pred_infer_logsdefault-tiny-034_ep011_pti01_tiny_yolo_leave-five-out_20181006101645',
    'pred_infer_logsdefault-tiny-038_ep007_pti01_tiny_yolo_leave-three-out_dicarding-training-frames_20181010135505',
    'pred_infer_logsdefault-tiny-041_ep009_pti01_tiny_yolo_dicarding-training-frames_20181010141205',

]
legends = [
    'High Bias',
    'Events out',
    '1 Camera out',
    '3 Camera out',
    '5 Camera out',
    '3 Camera out & Discard',
    'High Bias & Discard',
    'Tiny - High Bias',
    'Tiny - Events out',
    'Tiny - 1 Camera out',
    'Tiny - 3 Camera out',
    'Tiny - 5 Camera out',
    'Tiny - 3 Camera out & Discard',
    'Tiny - High Bias & Discard',
]

plot_missrate_fppi('compare_pti01_configurations', pti_tiny_line_style=True)

'''
PLOT:
Plotting for the same experiments of Figure 7.17 and 7.18.
Focus on PTI01.
'''

wanted_classes = [
    'person-no-ocl',
]
wanted_ious = [
    0.4
]
wanted_experiments = [
    'pred_infer_logsdefault-029_trained_weights_final.h5_pti01_yolo_dataset_by_event_20181101094129',
    'pred_infer_logsdefault-tiny-052_trained_weights_final.h5_pti01_tiny_yolo_tiny-dataset_by_event-oneclass_20181210203035',
]
legends = [
    'YOLOv3',
    'YOLOv3-tiny',
]

plot_missrate_fppi('yolov3_vs_yolov3tiny_pti01', legend_font_size=10)

'''
PLOT:
Plotting for the same experiments of Figure 7.17 and 7.18.
Focus on CALTECH.
'''

wanted_classes = [
    'person-no-ocl',
]
wanted_ious = [
    0.5
]
wanted_experiments = [
    'pred_infer_logscaltech_yolov3-001_ep001_caltech_yolo_caltech-new-config_new-dataset-without-single-pixel-annot_20181027234148',
    'pred_infer_logscaltech_default-001_ep002_caltech_tiny_yolo_tiny-caltech-new-config_20181028114020',
]
legends = [
    'YOLOv3',
    'YOLOv3-tiny',
]

plot_missrate_fppi('yolov3_vs_yolov3tiny_caltech', legend_font_size=10)






"""
############################
ROC curve
############################
"""



'''
PLOT:
Merging ROC curves
Focus on PTI01.
'''

wanted_classes = [
    'person-no-ocl',
]
wanted_ious = [
    0.4
]
wanted_experiments = [
    'pred_infer_logsdefault-029_trained_weights_final.h5_pti01_yolo_dataset_by_event_20181101094129',
    'pred_infer_logsdefault-tiny-052_trained_weights_final.h5_pti01_tiny_yolo_tiny-dataset_by_event-oneclass_20181210203035',
]
legends = [
    'YOLOv3',
    'YOLOv3-tiny',
]

plot_roc('yolov3_vs_yolov3tiny_pti01')

'''
PLOT:
Merging ROC curves
Focus on CALTECH.
'''

wanted_classes = [
    'person-no-ocl',
]
wanted_ious = [
    0.5
]
wanted_experiments = [
    'pred_infer_logscaltech_yolov3-001_ep001_caltech_yolo_caltech-new-config_new-dataset-without-single-pixel-annot_20181027234148',
    'pred_infer_logscaltech_default-001_ep002_caltech_tiny_yolo_tiny-caltech-new-config_20181028114020',
]
legends = [
    'YOLOv3',
    'YOLOv3-tiny',
]

plot_roc('yolov3_vs_yolov3tiny_caltech')
