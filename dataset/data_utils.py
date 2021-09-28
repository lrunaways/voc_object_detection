import torch
import torchvision
from torchvision.transforms import ToTensor

from voc_object_detection.dataset.ObjectDetectionDataset import ObjectDetectionDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_labels_map_dict(dataset):
    dataset_length = len(dataset)
    labels_map = {'background': 0}
    labels_set = set({})
    for i, instance in enumerate(dataset):
        if i % 100 == 0:
            print(f"{i}/{dataset_length}")
        for annotation in instance[1]['annotation']['object']:
            labels_set.add(annotation['name'])
    for i, label in enumerate(labels_set):
        labels_map[label] = i + 1
    return labels_map


def get_sizes_dict(dataset):
    sizes_dict = {
        'im_widths': [],
        'im_heights': [],
        'im_aspects': [],
        'im_areas': [],
        'bbox_widths': [],
        'bbox_heights': [],
        'bbox_aspects': [],
        'bbox_areas': [],
        'classes_histogram': {}
    }
    dataset_length = len(dataset)

    for i, instance in enumerate(dataset):
        instance = instance[1]['annotation']
        if i % 100 == 0:
            print(f"{i}/{dataset_length}")

        im_width = int(instance['size']['width'])
        im_height = int(instance['size']['height'])
        im_aspect = float(im_width) / im_height
        im_area = im_width * im_height
        sizes_dict['im_widths'].append(im_width)
        sizes_dict['im_heights'].append(im_height)
        sizes_dict['im_aspects'].append(im_aspect)
        sizes_dict['im_areas'].append(im_area)

        for annotation in instance['object']:
            class_name = annotation['name']
            bbox_annotation = annotation['bndbox']
            bbox_width = int(bbox_annotation['xmax']) - int(bbox_annotation['xmin'])
            bbox_height = int(bbox_annotation['ymax']) - int(bbox_annotation['ymin'])
            bbox_aspect = float(bbox_width) / bbox_height
            bbox_area = bbox_width * bbox_height

            sizes_dict['classes_histogram'][class_name] = \
                sizes_dict['classes_histogram'].get(class_name, 0) + 1
            sizes_dict['bbox_widths'].append(bbox_width)
            sizes_dict['bbox_heights'].append(bbox_height)
            sizes_dict['bbox_aspects'].append(bbox_aspect)
            sizes_dict['bbox_areas'].append(bbox_area)
    return sizes_dict


def get_voc_dataloader(image_set, size, batch_size, num_workers=2):
    voc_dataset = torchvision.datasets.VOCDetection(
        root="data",
        image_set=image_set,
        download=True,
        transform=ToTensor()
    )

    subset_indices_train = torch.randperm(len(voc_dataset))[:size]
    voc_dataset = torch.utils.data.Subset(voc_dataset, subset_indices_train)

    labels_map = get_labels_map_dict(voc_dataset)

    voc_dataset = ObjectDetectionDataset(voc_dataset, labels_map)

    voc_dataloader = torch.utils.data.DataLoader(
        voc_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn)
    return voc_dataloader

