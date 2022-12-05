import numpy as np
import jittor as jt
from jittor import nn

from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.utils.general import multi_apply,unmap
from jdet.utils.registry import HEADS,LOSSES,BOXES,build_from_cfg

# from jdet.ops.min_area_polygons import MinAreaPolygons
from jdet.ops.dcn_v1 import DeformConv
from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.box_ops import delta2bbox_rotated, rotated_box_to_poly
from jdet.models.boxes.anchor_target import images_to_levels,anchor_target
from jdet.models.boxes.point_generator import MlvlPointGenerator


def levels_to_images(mlvl_tensor, flatten=False):
    """Concat multi-level feature maps by image.
    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.
    Args:
        mlvl_tensor (list[jt.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
        flatten (bool, optional): if shape of mlvl_tensor is (N, C, H, W)
            set False, if shape of mlvl_tensor is  (N, H, W, C) set True.
    Returns:
        list[jt.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [jt.cat(item, 0) for item in batch_list]

@HEADS.register_module()
class RotatedRetinaHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=3,
                 point_feat_channels=256,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 transform_method='rotrect',
                 use_reassign=False,
                 topk=6,
                 anti_factor=0.75,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                    type='ConvexGIoULoss',
                    loss_weight=0.375),
                 loss_bbox_refine=dict(
                    type='ConvexGIoULoss',
                    loss_weight=1.0),
                 test_cfg=dict(
                    nms_pre=2000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms_rotated', iou_thr=0.1),
                    max_per_img=2000),
                train_cfg=dict(
                    init=dict(
                        assigner=dict(
                            type='ConvexAssigner',
                            scale=4,
                            pos_num=1),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False),
                    refine=dict(
                        assigner=dict(
                            type='MaxConvexIoUAssigner',
                            pos_iou_thr=0.4,
                            neg_iou_thr=0.3,
                            min_pos_iou=0,
                            ignore_iof_thr=-1),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False))):
        super(RotatedRetinaHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.point_feat_channels = point_feat_channels
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(
            self.point_strides, offset=0.)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.transform_method = transform_method
        self.use_reassign = use_reassign
        self.topk = topk
        self.anti_factor = anti_factor

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = jt.tensor(dcn_base_offset).view(1, -1, 1, 1)
    
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        self.loss_cls = build_from_cfg(loss_cls,LOSSES)
        self.loss_bbox_init = build_from_cfg(loss_bbox_init,LOSSES)
        self.loss_bbox_refine = build_from_cfg(loss_bbox_refine,LOSSES)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU()
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
        pts_out_dim = 2 * self.num_points

        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                               self.point_feat_channels,
                                               kernel_size=self.dcn_kernel, 
                                               padding=self.dcn_pad,
                                               deformable_groups=1)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                               self.point_feat_channels,
                                               kernel_size=self.dcn_kernel, 
                                               padding=self.dcn_pad,
                                               deformable_groups=1)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def init_weights(self):
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    # def points2rotrect(self, pts, y_first=True):
    #     """Convert points to oriented bboxes."""
    #     if y_first:
    #         pts = pts.reshape(-1, self.num_points, 2)
    #         pts_dy = pts[:, :, 0::2]
    #         pts_dx = pts[:, :, 1::2]
    #         pts = jt.cat([pts_dx, pts_dy],
    #                         dim=2).reshape(-1, 2 * self.num_points)
    #     if self.transform_method == 'rotrect':
    #         rotrect_pred = MinAreaPolygons(pts)
    #         return rotrect_pred
    #     else:
    #         raise NotImplementedError

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        pts_out_refine = pts_out_refine + pts_out_init.detach()

        return cls_out, pts_out_init, pts_out_refine

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl, _ in enumerate(self.point_strides):
            pts_lvl = []
            for i_img, _ in enumerate(center_list):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = jt.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = jt.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             overlaps,
                             stage='init',
                             unmap_outputs=True):
        """Single point target function."""
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight

        # convert gt from obb to poly
        gt_bboxes = rotated_box_to_poly(gt_bboxes,)

        assign_result = assigner.assign(proposals, gt_bboxes, overlaps,
                                        gt_bboxes_ignore,
                                        None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)
        num_valid_proposals = proposals.shape[0]
        bbox_gt = jt.zeros((num_valid_proposals, 8), dtype=proposals.dtype)
        pos_proposals = jt.zeros_like(proposals)
        proposals_weights = jt.zeros(num_valid_proposals, dtype=proposals.dtype)
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes).long()
        label_weights = jt.zeros(num_valid_proposals, dtype=proposals.dtype).float()

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, sampling_result)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.
        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple (list[Tensor]):
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of \
                    each level.
                - proposal_weights_list (list[Tensor]): Proposal weights of \
                    each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = jt.cat(proposals_list[i])
            valid_flag_list[i] = jt.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * len(proposals_list)
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_result) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)

        return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg, None)

    def get_cfa_targets(self,
                        proposals_list,
                        valid_flag_list,
                        gt_bboxes_list,
                        img_metas,
                        gt_bboxes_ignore_list=None,
                        gt_labels_list=None,
                        stage='init',
                        label_channels=1,
                        unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.
        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple:
                - all_labels (list[Tensor]): Labels of each level.
                - all_label_weights (list[Tensor]): Label weights of each \
                    level.
                - all_bbox_gt (list[Tensor]): Ground truth bbox of each level.
                - all_proposals (list[Tensor]): Proposals(points/bboxes) of \
                    each level.
                - all_proposal_weights (list[Tensor]): Proposal weights of \
                    each level.
                - pos_inds (list[Tensor]): Index of positive samples in all \
                    images.
                - gt_inds (list[Tensor]): Index of ground truth bbox in all \
                    images.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = jt.cat(proposals_list[i])
            valid_flag_list[i] = jt.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * len(proposals_list)
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_result) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        pos_inds = []
        # pos_gt_index = []
        for i, single_labels in enumerate(all_labels):
            pos_mask = (0 <= single_labels) & (
                single_labels < self.num_classes)
            pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))

        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]

        return (all_labels, all_label_weights, all_bbox_gt, all_proposals,
                all_proposal_weights, pos_inds, gt_inds)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, rbbox_gt_init, convex_weights_init,
                    rbbox_gt_refine, convex_weights_refine, stride,
                    num_total_samples_refine):
        """Single loss function."""
        normalize_term = self.point_base_scale * stride
        if self.use_reassign:
            rbbox_gt_init = rbbox_gt_init.reshape(-1, 8)
            convex_weights_init = convex_weights_init.reshape(-1)
            pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
            pos_ind_init = (convex_weights_init > 0).nonzero(
                as_tuple=False).reshape(-1)
            pts_pred_init_norm = pts_pred_init[pos_ind_init]
            rbbox_gt_init_norm = rbbox_gt_init[pos_ind_init]
            convex_weights_pos_init = convex_weights_init[pos_ind_init]
            loss_pts_init = self.loss_bbox_init(
                pts_pred_init_norm / normalize_term,
                rbbox_gt_init_norm / normalize_term, convex_weights_pos_init)
            return 0, loss_pts_init, 0
        else:
            rbbox_gt_init = rbbox_gt_init.reshape(-1, 8)
            convex_weights_init = convex_weights_init.reshape(-1)
            # init points loss
            pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
            pos_ind_init = (convex_weights_init > 0).nonzero(
                as_tuple=False).reshape(-1)
            pts_pred_init_norm = pts_pred_init[pos_ind_init]
            rbbox_gt_init_norm = rbbox_gt_init[pos_ind_init]
            convex_weights_pos_init = convex_weights_init[pos_ind_init]
            loss_pts_init = self.loss_bbox_init(
                pts_pred_init_norm / normalize_term,
                rbbox_gt_init_norm / normalize_term, convex_weights_pos_init)
            # refine points loss
            rbbox_gt_refine = rbbox_gt_refine.reshape(-1, 8)
            pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
            convex_weights_refine = convex_weights_refine.reshape(-1)
            pos_ind_refine = (convex_weights_refine > 0).nonzero(
                as_tuple=False).reshape(-1)
            pts_pred_refine_norm = pts_pred_refine[pos_ind_refine]
            rbbox_gt_refine_norm = rbbox_gt_refine[pos_ind_refine]
            convex_weights_pos_refine = convex_weights_refine[pos_ind_refine]
            loss_pts_refine = self.loss_bbox_refine(
                pts_pred_refine_norm / normalize_term,
                rbbox_gt_refine_norm / normalize_term,
                convex_weights_pos_refine)
            # classification loss
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=num_total_samples_refine)
            return loss_cls, loss_pts_init, loss_pts_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function of CFA head."""

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        device = cls_scores[0].device

        # target for initial stage
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if self.use_reassign:  # get num_proposal_each_lvl and lvl_num
            num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                        for featmap in cls_scores]
            num_level = len(featmap_sizes)
            assert num_level == len(pts_coordinate_preds_init)
        if self.train_cfg.init.assigner['type'] == 'ConvexAssigner':
            candidate_list = center_list
        else:
            raise NotImplementedError
        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, rbbox_gt_list_init, candidate_list_init, convex_weights_list_init,
         num_total_pos_init, num_total_neg_init, _) = cls_reg_targets_init
        # target for refinement stage
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(
                    points_preds_init_.shape[0], -1,
                    *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(
                    0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(
                    points_center +
                    points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)
        if self.use_reassign:
            cls_reg_targets_refine = self.get_cfa_targets(
                points_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                stage='refine',
                label_channels=label_channels)
            (labels_list, label_weights_list, rbbox_gt_list_refine, _,
             convex_weights_list_refine, pos_inds_list_refine,
             pos_gt_index_list_refine) = cls_reg_targets_refine
            cls_scores = levels_to_images(cls_scores)
            cls_scores = [
                item.reshape(-1, self.cls_out_channels) for item in cls_scores
            ]
            pts_coordinate_preds_init_cfa = levels_to_images(
                pts_coordinate_preds_init, flatten=True)
            pts_coordinate_preds_init_cfa = [
                item.reshape(-1, 2 * self.num_points)
                for item in pts_coordinate_preds_init_cfa
            ]
            pts_coordinate_preds_refine = levels_to_images(
                pts_coordinate_preds_refine, flatten=True)
            pts_coordinate_preds_refine = [
                item.reshape(-1, 2 * self.num_points)
                for item in pts_coordinate_preds_refine
            ]
            with jt.no_grad():
                pos_losses_list, = multi_apply(
                    self.get_pos_loss, cls_scores,
                    pts_coordinate_preds_init_cfa, labels_list,
                    rbbox_gt_list_refine, label_weights_list,
                    convex_weights_list_refine, pos_inds_list_refine)
                labels_list, label_weights_list, convex_weights_list_refine, \
                    num_pos, pos_normalize_term = multi_apply(
                        self.reassign,
                        pos_losses_list,
                        labels_list,
                        label_weights_list,
                        pts_coordinate_preds_init_cfa,
                        convex_weights_list_refine,
                        gt_bboxes,
                        pos_inds_list_refine,
                        pos_gt_index_list_refine,
                        num_proposals_each_level=num_proposals_each_level,
                        num_level=num_level
                    )
                num_pos = sum(num_pos)
            # convert all tensor list to a flatten tensor
            cls_scores = jt.cat(cls_scores, 0).view(-1,
                                                       cls_scores[0].size(-1))
            pts_preds_refine = jt.cat(pts_coordinate_preds_refine, 0).view(
                -1, pts_coordinate_preds_refine[0].size(-1))
            labels = jt.cat(labels_list, 0).view(-1)
            labels_weight = jt.cat(label_weights_list, 0).view(-1)
            rbbox_gt_refine = jt.cat(rbbox_gt_list_refine, 0).view(
                -1, rbbox_gt_list_refine[0].size(-1))
            convex_weights_refine = jt.cat(convex_weights_list_refine,
                                              0).view(-1)
            pos_normalize_term = jt.cat(pos_normalize_term, 0).reshape(-1)
            pos_inds_flatten = ((0 <= labels) &
                                (labels < self.num_classes)).nonzero(
                                    as_tuple=False).reshape(-1)
            assert len(pos_normalize_term) == len(pos_inds_flatten)
            if num_pos:
                losses_cls = self.loss_cls(
                    cls_scores, labels, labels_weight, avg_factor=num_pos)
                pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
                pos_rbbox_gt_refine = rbbox_gt_refine[pos_inds_flatten]
                pos_convex_weights_refine = convex_weights_refine[
                    pos_inds_flatten]
                losses_pts_refine = self.loss_bbox_refine(
                    pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                    pos_rbbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                    pos_convex_weights_refine)
            else:
                losses_cls = cls_scores.sum() * 0
                losses_pts_refine = pts_preds_refine.sum() * 0
            None_list = [None] * num_level
            _, losses_pts_init, _ = multi_apply(
                self.loss_single,
                None_list,
                pts_coordinate_preds_init,
                None_list,
                None_list,
                None_list,
                rbbox_gt_list_init,
                convex_weights_list_init,
                None_list,
                None_list,
                self.point_strides,
                num_total_samples_refine=None,
            )
            loss_dict_all = {
                'loss_cls': losses_cls,
                'loss_pts_init': losses_pts_init,
                'loss_pts_refine': losses_pts_refine
            }
            return loss_dict_all
        else:
            cls_reg_targets_refine = self.get_targets(
                points_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                stage='refine',
                label_channels=label_channels)
            (labels_list, label_weights_list, rbbox_gt_list_refine,
             candidate_list_refine, convex_weights_list_refine,
             num_total_pos_refine, num_total_neg_refine,
             _) = cls_reg_targets_refine
            num_total_samples_refine = (
                num_total_pos_refine + num_total_neg_refine
                if self.sampling else num_total_pos_refine)

            losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
                self.loss_single,
                cls_scores,
                pts_coordinate_preds_init,
                pts_coordinate_preds_refine,
                labels_list,
                label_weights_list,
                rbbox_gt_list_init,
                convex_weights_list_init,
                rbbox_gt_list_refine,
                convex_weights_list_refine,
                self.point_strides,
                num_total_samples_refine=num_total_samples_refine)
            loss_dict_all = {
                'loss_cls': losses_cls,
                'loss_pts_init': losses_pts_init,
                'loss_pts_refine': losses_pts_refine
            }
            return loss_dict_all

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   rescale=True):
        assert len(cls_scores) == len(bbox_preds)
        cfg = self.test_cfg.copy()

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        num_levels = len(cls_scores)
        anchor_list, _ = self.get_init_anchors(featmap_sizes, img_metas)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, 
                                               anchor_list[img_id], img_shape,
                                               scale_factor, cfg, rescale)

            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            # anchors = rect2rbox(anchors)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores = scores.max(dim=1)
                else:
                    max_scores = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = jt.contrib.concat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= scale_factor
        mlvl_scores = jt.contrib.concat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = jt.zeros((mlvl_scores.shape[0], 1),dtype=mlvl_scores.dtype)
            mlvl_scores = jt.contrib.concat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes,
                                                        mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

    
    def parse_targets(self,targets,is_train=True):
        img_metas = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_labels = []

        for target in targets:
            if is_train:
                gt_bboxes.append(target["rboxes"])
                gt_labels.append(target["labels"])
                gt_bboxes_ignore.append(target["rboxes_ignore"])
            img_metas.append(dict(
                img_shape=target["img_size"][::-1],
                scale_factor=target["scale_factor"],
                pad_shape = target["pad_shape"]
            ))
        if not is_train:
            return img_metas
        return gt_bboxes,gt_labels,img_metas,gt_bboxes_ignore

    def execute(self, feats,targets):
        outs = multi_apply(self.forward_single, feats)
        if self.is_training():
            return self.loss(*outs,*self.parse_targets(targets))
        else:
            return self.get_bboxes(*outs,self.parse_targets(targets,is_train=False))
