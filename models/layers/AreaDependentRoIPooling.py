import numpy as np
import torch
from torch import nn, Tensor
import torchvision


class AreaDependentRoIPooling(nn.Module):
    """
    See :func:`roi_pool`.
    """
    def __init__(self, featmap_names, output_size, spatial_scale, sampling_ratio, small_obj_threshold=32*32):
        super(AreaDependentRoIPooling, self).__init__()
        self.featmap_names = featmap_names
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.small_obj_roi_threshold = small_obj_threshold * spatial_scale
        self.roi_pool = torchvision.ops.RoIPool(output_size=output_size, spatial_scale=spatial_scale)
        self.roi_align = torchvision.ops.RoIAlign(output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)

    # get back data types
    def forward(self, input: Tensor, rois: Tensor, image_sizes=None) -> Tensor:
      if self.featmap_names is not None:
        input = input[self.featmap_names[0]]

      areas = torch.cat([(x[:, 2] - x[:, 0])*(x[:, 3] - x[:, 1]) for x in rois]).view(-1, 1, 1, 1)

      roi_pooled = self.roi_pool(input, rois)
      roi_aligned = self.roi_align(input, rois)
      roi_adp = torch.where(areas <= self.small_obj_roi_threshold, roi_aligned, roi_pooled)
      return roi_adp

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', small_obj_threshold=' + str(self.small_obj_threshold)
        tmpstr += ')'
        return tmpstr


if __name__ == "__main__":
    # test module
    output_size = 3
    spatial_scale = 1  # 1./16.
    sampling_ratio = 2

    adp = AreaDependentRoIPooling(None, output_size, spatial_scale, sampling_ratio, small_obj_threshold=7)
    roi_pool = torchvision.ops.RoIPool(output_size=output_size, spatial_scale=spatial_scale)
    roi_align = torchvision.ops.RoIAlign(output_size=output_size, spatial_scale=spatial_scale,
                                         sampling_ratio=sampling_ratio)

    im = torch.Tensor(np.arange(2 * 8 * 8).reshape((2, 1, 8, 8)))

    rois = [torch.Tensor(
        [
            [0, 0, 3, 2],
            [0, 0, 3, 3],
            [0, 0, 6, 6],
            [1, 1, 5, 5]
        ]
    )] * 2

    print('roi_pool', roi_pool(im, rois))
    print('roi_align', roi_align(im, rois))
    print('adp', adp(im, rois))
    # TODO: assert adp(im, rois) == ...
