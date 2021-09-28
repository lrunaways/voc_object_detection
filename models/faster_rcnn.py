import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from voc_object_detection.models.layers.AreaDependentRoIPooling import AreaDependentRoIPooling


def get_faster_rcnn_model(num_classes=21,
                          anchor_sizes=((128, 256, 512),),
                          anchor_aspect_ratios=((0.5, 1.0, 2.0),),
                          pooler='area_dependent',
                          pooler_out_size=3,
                          max_size=1024):
    available_poolers = ['area_dependent', 'roi_pooling', 'roi_align']
    backbone = torchvision.models.vgg16(pretrained=True).features
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=anchor_aspect_ratios)

    # при инициализации все равно создается пулер - создадим нужного размера
    roi_pooler_temp = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=pooler_out_size,
                                                         sampling_ratio=2)

    # put the pieces together inside a FasterRCNN models
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler_temp,
                       max_size=max_size)

    if pooler in available_poolers:
        print(f"Using {pooler} pooler")
        if pooler == 'area_dependent':
            roi_pooler = AreaDependentRoIPooling(featmap_names=['0'], output_size=pooler_out_size,
                                                 spatial_scale=1./32., sampling_ratio=2, small_obj_threshold=32**2)
        elif pooler == 'roi_pooling':
            roi_pooler = AreaDependentRoIPooling(featmap_names=['0'], output_size=pooler_out_size,
                                                 spatial_scale=1./32., sampling_ratio=2, small_obj_threshold=0)
        elif pooler == 'roi_align':
            roi_pooler = AreaDependentRoIPooling(featmap_names=['0'], output_size=pooler_out_size,
                                                 spatial_scale=1./32., sampling_ratio=2, small_obj_threshold=1024**2)
        model.roi_heads.box_roi_pool = roi_pooler
    return model


if __name__ == '__main__':
    model = get_faster_rcnn_model(pooler='area_dependent')
    print(model)
