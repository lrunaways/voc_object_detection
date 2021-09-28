import torch
import torchvision


def calc_tp_fp_fn_for_example(true_boxes, pred_boxes, iou_threshold, device):
  # pred_boxes are sorted by score!
  TP_FP_map = torch.zeros(len(pred_boxes), dtype=torch.bool, device=device)  # True для TP, False для FP
  GT_detected = torch.zeros(len(true_boxes), dtype=torch.bool, device=device)  # GT_detected: # True для детектированных
                                                                               # ббоксов, False для FN
  if len(true_boxes) and len(pred_boxes):
    #TODO: оптимизировать, чтобы не вычислять iou для каждого порога
    iou_matrix = torchvision.ops.box_iou(pred_boxes, true_boxes)
    iou_matrix = iou_matrix > iou_threshold
    GT_detected, TP_idx = iou_matrix.max(axis=0) # находим есть ли для gt ббоксы > iou_threshold
    for i, detected in enumerate(GT_detected):
      if detected:
        TP_FP_map[TP_idx[i]] = True
  return TP_FP_map, GT_detected


if __name__=='__main__':
    from voc_object_detection.metrics.f1 import calc_f1
    import matplotlib.pyplot as plt

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image = torch.zeros((3, 500, 500), dtype=torch.uint8)
    true_boxes = [
        [0, 100, 100, 200],
        [400, 400, 500, 500],
        [300, 300, 450, 450],
        [200, 200, 300, 300],
        [200, 0, 300, 100],
    ]
    pred_boxes = [
        [0, 140, 80, 200],
        [380, 380, 480, 480],  # IoU с [400, 400, 500, 500] = 0.47
        [0, 100, 120, 220],
        [300, 300, 500, 500],
        [200, 20, 300, 120],
        [10, 10, 50, 50]
    ]

    true_boxes = torch.Tensor(true_boxes)
    pred_boxes = torch.Tensor(pred_boxes)

    colors_true = [(60, 200, 50)] * len(true_boxes)
    colors_pred = [(200, 30, 30)] * len(pred_boxes)

    TP_FP_map, GT_detected = calc_tp_fp_fn_for_example(true_boxes, pred_boxes, iou_threshold=0.5, device=device)
    print('F1 score:', calc_f1(TP_FP_map, GT_detected))

    labels_pred = ['TP+' if x else 'FP-' for x in TP_FP_map]
    labels_true = ['Detected' if x else 'FN' for x in GT_detected]

    all_boxes = torch.cat([true_boxes, pred_boxes], axis=0)
    all_labels = labels_true + labels_pred
    all_colors = colors_true + colors_pred

    bboxed_image = torchvision.utils.draw_bounding_boxes(image, all_boxes, all_labels, colors=all_colors, width=3)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(False)
    ax.imshow(bboxed_image.permute(1, 2, 0))
    plt.show()