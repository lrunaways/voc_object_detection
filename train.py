from voc_object_detection.dataset.data_utils import get_voc_dataloader
from voc_object_detection.models.faster_rcnn import get_faster_rcnn_model
from voc_object_detection.training.training_loop import training_loop


if __name__ == '__main__':
    train_dataloader = get_voc_dataloader(image_set='train', size=500, batch_size=2)
    val_dataloader = get_voc_dataloader(image_set='val', size=200, batch_size=1)

    labels_map = train_dataloader.dataset.labels_map  # маппит имена в целые лейблы классов
    assert len(labels_map) == 20 + 1  # 20 класов + 'background'

    model = get_faster_rcnn_model(num_classes=len(labels_map),
                                  anchor_sizes=((128, 256, 512),),
                                  anchor_aspect_ratios=((0.5, 1.0, 2.0),),
                                  pooler='area_dependent',
                                  pooler_out_size=3)

    history_dict = training_loop(model, train_dataloader, val_dataloader, n_epochs=5, labels_map=labels_map)
