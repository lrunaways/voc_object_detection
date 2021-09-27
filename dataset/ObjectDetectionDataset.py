import torch
from torch.utils.data import Dataset


class ObjectDetectionDataset(Dataset):
    def __init__(self, dataset, labels_map):
        self.dataset = dataset
        self.labels_map = labels_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, annotation = self.dataset[idx]
        annotation = annotation['annotation']

        im_height = int(annotation['size']['height'])
        im_width = int(annotation['size']['width'])

        boxes = []
        labels = []
        labels_named = []
        areas = []
        for i_box, obj in enumerate(annotation['object']):
            bbox = [
                # i_box,
                int(obj['bndbox']['xmin']),
                int(obj['bndbox']['ymin']),
                int(obj['bndbox']['xmax']),
                int(obj['bndbox']['ymax'])
            ]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            boxes.append(bbox)
            labels.append(self.labels_map[obj['name']])
            labels_named.append(obj['name'])
            areas.append(area)

        target = {'im_height': torch.as_tensor([im_height], dtype=torch.int64),
                  'im_width': torch.as_tensor([im_width], dtype=torch.int64),
                  'boxes': torch.as_tensor(boxes, dtype=torch.int64),
                  'labels': torch.as_tensor(labels, dtype=torch.int64),
                  'area': torch.as_tensor(areas, dtype=torch.int64)}
        return image, target
