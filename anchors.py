import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None,
                 ratios=None, scales=None,
                 image_shape=None,
                 squeeze=True,
                 exp_base=2):
        super(Anchors, self).__init__()

        self.exp_base=exp_base
        self.squeeze = squeeze
        if pyramid_levels is None:
            self._pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self._pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [self.exp_base** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [self.exp_base** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)])
        if image_shape is not None:
            self.image_shape = image_shape[-2:]
            self.pyramid_shapes = self.compute_pyramid_shapes(self.image_shape, self.pyramid_levels,
                                                              exp_base=self.exp_base)
        else:
            self.pyramid_shapes = None
        # print("self.pyramid_levels @ init", self.pyramid_levels)

    @property
    def pyramid_levels(self):
        return self._pyramid_levels

    @classmethod
    def compute_pyramid_shapes(cls, image_shape, pyramid_levels, exp_base=2):
        image_shape = image_shape[-2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + exp_base**x - 1) // (exp_base**x) for x in pyramid_levels]
        return image_shapes

    def compute_anchors(self):
        #all_anchors = np.zeros((0, 4)).astype(np.float32)
        all_anchors = []
        # print("self.pyramid_levels @ compute_anchors", self.pyramid_levels)

        if self.squeeze:
            for idx, _ in enumerate(self.pyramid_levels):
                anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
                # print("anchors: idx", idx, anchors.shape)
                shifted_anchors = shift(self.pyramid_shapes[idx], self.strides[idx], anchors)
                all_anchors.append(shifted_anchors)
            all_anchors = np.concatenate(all_anchors, axis=0)
            all_anchors = np.expand_dims(all_anchors, axis=0)
            return torch.from_numpy(all_anchors.astype(np.float32))#.cuda()
        else:
            for idx, _ in enumerate(self.pyramid_levels):
                anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
                shifted_anchors = shift_keep_shape(self.pyramid_shapes[idx], self.strides[idx], anchors)
                all_anchors.append(shifted_anchors)

            all_anchors = [torch.from_numpy(an.astype(np.float32)) for an in all_anchors]
            print([x.shape for x in all_anchors])
            return all_anchors

    def anchors(self, image_shape=None):
        if not hasattr(self, '_anchors_'):
            self._anchors_ = self.compute_anchors()
        return self._anchors_

    def forward(self, image):
        if self.pyramid_shapes is None:
            self.pyramid_shapes = self.compute_pyramid_shapes(image.shape, self.pyramid_levels)
        return self.anchors(image.shape[-2:])
        # compute anchors over all pyramid levels

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift_keep_shape(shape, stride, anchors):
    anch_width = anchors[:,2]
    anch_height = anchors[:,3]

    #shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    #shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    print(shift_x.min(), shift_x.max())
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anch_x1 = (shift_x - anch_width[...,np.newaxis][...,np.newaxis])
    anch_x2 = (shift_x + anch_width[...,np.newaxis][...,np.newaxis])
    anch_y1 = (shift_y - anch_height[...,np.newaxis][...,np.newaxis])
    anch_y2 = (shift_y + anch_height[...,np.newaxis][...,np.newaxis])
    
    shifted_anchors = np.stack([anch_x1, anch_y1, anch_x2, anch_y2])
    return shifted_anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

