import numpy as np
import torch

from metrics.metric_utils import filter_by_area, filter_by_label, nms_dict
from metrics.calc_errors import calc_tp_fp_fn_for_example


def calc_f1(tp_fp_map, gt_detected):
    tp = torch.sum(tp_fp_map)
    fp = torch.sum(~tp_fp_map)
    fn = torch.sum(~gt_detected)
    return 2*tp/(2*tp + fp + fn)


def calc_mean_f1(metrics_dict):
    means_of_classes_dict = {}
    for key in metrics_dict:
        if key.startswith('iou_threshold'):
            for class_label in metrics_dict[key]:
                if means_of_classes_dict.get(class_label, None) is None:
                    means_of_classes_dict[class_label] = []
                means_of_classes_dict[class_label].append(metrics_dict[key][class_label])
    for key in means_of_classes_dict:
        means_of_classes_dict[key] = sum(means_of_classes_dict[key]) / len(means_of_classes_dict[key])
    total_score_for_classes = np.array([means_of_classes_dict[key].item() for key in means_of_classes_dict])
    total_score_for_classes = total_score_for_classes[~np.isnan(total_score_for_classes)]
    means_of_classes_dict['total'] = total_score_for_classes.mean()
    return means_of_classes_dict


def calc_f1_full(list_of_true_dict, list_of_pred_dict, class_labels, labels_to_names_map,
                 iou_thresholds, device, score_threshold=0.0, nms_iou=0.5, area_min=0, area_max=2**32):
    assert len(list_of_pred_dict) == len(list_of_true_dict)
    class_labels = class_labels[1:] if class_labels[0] == 0 else class_labels # skip background class
    metrics_dict = {'area_min': area_min, 'area_max': area_max}
    for iou_threshold in iou_thresholds:
        iou_threshold_name = f"iou_threshold_{iou_threshold:.2f}"
        for class_label in class_labels:
            tp_fp_map, gt_detected = [], []
            for i in range(len(list_of_pred_dict)):
                true_dict, pred_dict = list_of_true_dict[i], list_of_pred_dict[i]

                #TODO: выделить фильтрацию и nms в отдельную функцию
                true_dict, pred_dict = (filter_by_label(true_dict, class_label),
                                        filter_by_label(pred_dict, class_label))
                true_dict, pred_dict = (filter_by_area(true_dict, area_min, area_max),
                                        filter_by_area(pred_dict, area_min, area_max))
                pred_dict = nms_dict(pred_dict, nms_iou)

                tp_fp_map_example, gt_detected_example = calc_tp_fp_fn_for_example(true_dict['boxes'], pred_dict['boxes'],
                                                                                   iou_threshold, device)
                tp_fp_map.append(tp_fp_map_example)
                gt_detected.append(gt_detected_example)
            tp_fp_map = torch.cat(tp_fp_map)
            gt_detected = torch.cat(gt_detected)
            f1_score = calc_f1(tp_fp_map, gt_detected)
            if metrics_dict.get(iou_threshold_name, None) is None:
                metrics_dict[iou_threshold_name] = metrics_dict.get(iou_threshold_name, {})
            metrics_dict[iou_threshold_name][labels_to_names_map[class_label]] = f1_score
    metrics_dict['F1_mean_by_class'] = calc_mean_f1(metrics_dict)
    return metrics_dict


def evaluate_f1(model, data_loader, device, class_labels, labels_to_names_map, iou_thresholds=np.arange(0.5, 0.95, 0.05), dataset_name=""):
    print(f'Evaluating on {dataset_name}...')
    y_preds = []
    y_trues = []
    metrics = {}

    model.eval()
    for i, (images, targets) in enumerate(data_loader):
        if i % 50 == 0:
            print(f"{i}/{len(data_loader)}")
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred = model(images)
        y_preds.extend(pred)
        y_trues.extend(targets)

    metrics['small_objects'] = calc_f1_full(y_trues, y_preds, class_labels, labels_to_names_map, iou_thresholds,
                                            area_min=0, area_max=32 * 32)
    metrics['large_objects'] = calc_f1_full(y_trues, y_preds, class_labels, labels_to_names_map, iou_thresholds,
                                            area_min=96 * 96, area_max=1000 * 1000)

    print(f"{dataset_name} F1:    small objects: {metrics['small_objects']['F1_mean_by_class']['total']} \
    large objects: {metrics['large_objects']['F1_mean_by_class']['total']}")
    return metrics
