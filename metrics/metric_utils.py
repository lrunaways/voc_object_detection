import torch
import torchvision

def get_filtered_dict(data_dict, filter_map):
    proc_data_dict = data_dict.copy()
    for key in data_dict:
        filtered_vals = [x for x, filter in zip(data_dict[key], filter_map) if filter]
        if filtered_vals:
            filtered_vals = torch.stack(filtered_vals, axis=0)
        proc_data_dict[key] = filtered_vals
    return proc_data_dict


def filter_by_area(data_dict, min_area, max_area):
    filter_map = [True if min_area <= (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) <= max_area else False
                  for bbox in data_dict['boxes']]
    proc_data_dict = get_filtered_dict(data_dict, filter_map)
    return proc_data_dict


def filter_by_label(data_dict, class_label):
    filter_map = [True if label == class_label else False for label in data_dict['labels']]
    proc_data_dict = get_filtered_dict(data_dict, filter_map)
    return proc_data_dict


def nms_dict(data_dict, iou_threshold):
    proc_data_dict = data_dict.copy()
    if len(proc_data_dict['boxes']):
        idx_nms = torchvision.ops.nms(proc_data_dict['boxes'], proc_data_dict['scores'], iou_threshold=iou_threshold)
        for key in proc_data_dict:
            proc_data_dict[key] = proc_data_dict[key][idx_nms]
    return proc_data_dict



