"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.cluster import DBSCAN
from opencood.utils.transformation_utils import x_to_world,x1_to_x2



def to_polygon(box):
    return Polygon([(box[i, 0], box[i, 1]) for i in range(4)])

def compute_self_iou_mat(boxes, dist_threshold=10.0):
    centers = boxes.mean(axis=1)
    dist_mat = cdist(centers, centers)
    iou_mat = np.zeros_like(dist_mat)
    np.fill_diagonal(iou_mat, 1.0)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if dist_mat[i, j] < dist_threshold:
                polygon1 = to_polygon(boxes[i])
                polygon2 = to_polygon(boxes[j])
                iou = polygon1.intersection(polygon2).area / polygon1.union(
                    polygon2).area
                if iou > iou_mat[i, j]:
                    iou_mat[i, j] = iou
                    iou_mat[j, i] = iou
    return iou_mat

def compute_cross_iou_mat(ego_preds,cav_preds, dist_threshold=10.0):
    ego_centers = ego_preds.mean(axis=1)
    cav_centers = cav_preds.mean(axis=1)
    dist_mat = cdist(ego_centers, cav_centers)
    iou_mat = np.zeros_like(dist_mat)
    #np.fill_diagonal(iou_mat, 1.0)
    for i in range(len(ego_preds)):
        for j in range(len(cav_preds)):
            if dist_mat[i, j] < dist_threshold:
                polygon1 = to_polygon(ego_preds[i])
                polygon2 = to_polygon(cav_preds[j])
                iou = polygon1.intersection(polygon2).area / polygon1.union(
                    polygon2).area
                if iou > iou_mat[i, j]:
                    iou_mat[i, j] = iou
                    #iou_mat[j, i] = iou
    return iou_mat

class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh']
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2

        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl':
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = \
            box_utils.boxes_to_corners_3d(gt_box_center_valid,
                                          self.params['order'])
        # (H*W*anchor_num, 8, 3)
        anchors_corner = \
            box_utils.boxes_to_corners_3d(anchors,
                                          order=self.params['order'])
        # (H*W*anchor_num, 4)
        anchors_standup_2d = \
            box_utils.corner2d_to_standup_box(anchors_corner)
        # (n, 4)
        gt_standup_2d = \
            box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        # (H*W*anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold'])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []

        vehicle_location = []

        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            cur_location = torch.tensor([[0.,0.,0.]])

            trans_cur_location = box_utils.project_points_by_matrix_torch(cur_location,
                                            transformation_matrix.cpu())
            vehicle_location.append(trans_cur_location)
            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['psm']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['rm']

            # convert regression map back to bounding box
            # (N, W*L*anchor_num, 7)
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None


        # pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores = pred_box2d_list[:, -1]
        # pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # #icra
        # preds = pred_box3d_tensor.cpu()
        # probs = scores.cpu()
        # iou_mat = compute_self_iou_mat(preds)
        # clusters, visited, selected = [], [], []
        # for idx, ious in enumerate(iou_mat):
        #     if idx in visited:
        #         continue
        #     neighbor_idxs = np.nonzero(ious)[0]
        #     clusters.append(neighbor_idxs)
        #     visited.extend(neighbor_idxs)
        # for cluster in clusters:
        #     sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
        #     sub_probs = probs[cluster]
        #     values = sub_iou_mat.dot(sub_probs)
        #     soft_bools = softmax(values / 1e-6)
        #     bools = soft_bools > 0.5
        #     selected.extend(cluster[bools])
        # pred_box3d_tensor = preds[selected]
        # scores = probs[selected]
        # print('icra:',len(scores))
        # sorted_list = sorted(selected)
        # print('icra selected:',sorted_list)
        # #24
        # #pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores = pred_box2d_list[:, -1]
        # pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # # nms
        # keep_index = box_utils.nms_rotated(pred_box3d_tensor,
        #                                    scores,
        #                                    self.params['nms_thresh']
        #                                    )

        # pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # # select cooresponding score
        # scores = scores[keep_index]

        # # filter out the prediction out of the range.
        # mask = \
        #     box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        # pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        # scores = scores[mask]
        # print('nms:',len(scores))
        # sorted_list = sorted(keep_index)
        # print('nms selected:',sorted_list)
        #24

        ####ours

        ###icra
        # total_v1 = []
        # total_v2 = []
        # for v1,v2 in zip(pred_box2d_list,pred_box3d_list):   
        #     probs = v1[:, -1].cpu()
        #     preds = v2.cpu()
            
        #     iou_mat = compute_self_iou_mat(preds)
        #     clusters, visited, selected = [], [], []
        #     for idx, ious in enumerate(iou_mat):
        #         if idx in visited:
        #             continue
        #         neighbor_idxs = np.nonzero(ious)[0]
        #         clusters.append(neighbor_idxs)
        #         visited.extend(neighbor_idxs)
        #     for cluster in clusters:
        #         sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
        #         sub_probs = probs[cluster]
        #         values = sub_iou_mat.dot(sub_probs)
        #         soft_bools = softmax(values / 1e-6)
        #         bools = soft_bools > 0.5
        #         selected.extend(cluster[bools])
        #     pred_box3d_tensor = preds[selected]
        #     scores = probs[selected]

        #     total_v1.append(pred_box3d_tensor)
        #     total_v2.append(scores)
#nms
# The Average Precision at IOU 0.3 is 0.846, The Average Precision at IOU 0.5 is 0.842, The Average Precision at IOU 0.7 is 0.73
#psa
# The Average Precision at IOU 0.3 is 0.846, The Average Precision at IOU 0.5 is 0.842, The Average Precision at IOU 0.7 is 0.750
        ###nms
        

        # total_v1 = []
        # total_v2 = []
        # for v1,v2 in zip(pred_box2d_list,pred_box3d_list):   
        #     scores = v1[:, -1].cpu()
        #     pred_box3d_tensor = v2.cpu()
        #     keep_index = box_utils.nms_rotated(pred_box3d_tensor,
        #                                     scores,
        #                                     self.params['nms_thresh']
        #                                     )

        #     pred_box3d_tensor = pred_box3d_tensor[keep_index]

        #     # select cooresponding score
        #     scores = scores[keep_index]

        #     # filter out the prediction out of the range.
        #     mask = \
        #         box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        #     pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        #     scores = scores[mask]

        #     total_v1.append(pred_box3d_tensor)
        #     total_v2.append(scores)
        # #vehicle_location
        # for i in range(len(total_v1)):
        #     preds = total_v1[i]
        #     centers = preds.mean(axis=1)[:,:2]
        #     probs = total_v2[i]
        #     vehicle_point = vehicle_location[i][:,:2]
        #     distance = torch.sqrt(torch.sum((centers - vehicle_point) ** 2, axis=1))
        #     for indx in range(len(distance)):
        #         if distance[indx] > 20:
        #             total_v2[i][indx] = total_v2[i][indx] * 0.7
        #     indx_list = torch.argsort(distance)


        # cav_num = len(total_v1) - 1
        # if cav_num > 1:
        #     cav_v1 = torch.cat(total_v1[1:],dim=0)
        #     cav_v2 = torch.cat(total_v2[1:],dim=0)
        #     total_v1 = [total_v1[0],cav_v1]
        #     total_v2 = [total_v2[0],cav_v2]
            

        # total_ego_selected,pred_box3d_tensor, scores = [], [], []
        # for i in range(1,len(total_v1)):
        #     ego_preds = total_v1[0].cpu()
        #     ego_probs = total_v2[0].cpu()

        #     cav_preds = total_v1[i].cpu()
        #     cav_probs = total_v2[i].cpu()

            
        #     cross_iou_mat = compute_cross_iou_mat(ego_preds,cav_preds)
        #     clusters, cav_visited, ego_visited, ego_selected, cav_selected = [], [], [], [], []
        #     for idx, ious in enumerate(cross_iou_mat):
        #         # if idx in visited:
        #         #     continue
        #         neighbor_idxs = np.nonzero(ious)[0]
        #         real_idxs = np.concatenate((np.array([idx]),neighbor_idxs))
        #         clusters.append(real_idxs)
        #         cav_visited.extend(neighbor_idxs)
        #     for cluster in clusters:
        #         if len(cluster) == 1:
        #             #ego_selected.extend(cluster)
        #             continue
        #         v1 = cluster[0].repeat(len(cluster))


        #         sub_iou_mat = np.zeros((len(cluster),len(cluster)))
        #         np.fill_diagonal(sub_iou_mat, 1.0)
        #         for index in range(1,len(cluster)):
        #             sub_iou_mat[index][len(cluster)-1-index] = cross_iou_mat[cluster[0]][cluster[index]]
        #             sub_iou_mat[len(cluster)-1-index][index] = cross_iou_mat[cluster[0]][cluster[index]]
        #         np.fill_diagonal(sub_iou_mat, 1.0)
        #         #sub_iou_mat = cross_iou_mat[np.ix_(v1, cluster)]
        #         sub_probs_ego = ego_probs[cluster[0]]
        #         sub_probs_cav = cav_probs[cluster[1:]]
        #         sub_probs = np.concatenate(([sub_probs_ego],sub_probs_cav))
        #         values = sub_iou_mat.dot(sub_probs)
        #         soft_bools = softmax(values / 1e-6)
        #         bools = soft_bools > 0.5

        #         if bools[0] == True:
        #             v1 = int(cluster[0]) 
        #             ego_selected.append(int(cluster[0]))
        #             ego_visited.append(int(cluster[0]))
        #         else:
        #             for index in range(1,len(cluster)):
        #                 if bools[index] == True:
        #                     cav_selected.append(int(cluster[index]))
        #                     ego_visited.append(int(cluster[0]))

            
        #     for item in ego_selected:
        #         if item in total_ego_selected:
        #             ego_selected.remove(item)
        #     cav_selected = set(cav_selected)
        #     cav_selected = list(cav_selected)
        #     pred_box3d_tensor.extend(ego_preds[ego_selected])
        #     pred_box3d_tensor.extend(cav_preds[cav_selected])
        #     # v1 = ego_probs[ego_selected]
        #     # v2 = np.exp(v1)/np.exp(np.ones_like(v1))
        #     # v1 = ego_probs[ego_selected]
        #     # v2 = np.clip(v1,0.9,1)

        #     scores.extend(ego_probs[ego_selected])
        #     # v3 = cav_probs[cav_selected]
        #     # v4 = np.exp(v3)/np.exp(np.ones_like(v3))
        #     # v3 = cav_probs[cav_selected]
        #     # v4 = np.clip(v3,0.9,1)
        #     scores.extend(cav_probs[cav_selected])
        #     total_ego_selected.extend(ego_selected)
        #     # if i != 0:
        #     cav_index = range(len(cav_preds))



        #     ego_by_cav_list = []
        #     for item in cav_index:
        #         if item not in cav_visited:
        #             v1 = cav_preds[item]
        #             v1 = v1[np.newaxis,:,:]
        #             centers = v1.mean(axis=1)[:,:2]
        #             vehicle_point = vehicle_location[0][:,:2]
        #             distance = torch.sqrt(torch.sum((centers - vehicle_point) ** 2, axis=1))
        #             if distance < 1:
        #                 ego_by_cav_list.append(item)
        #                 continue

        #             pred_box3d_tensor.append(cav_preds[item])
        #             scores.append(cav_probs[item])
            
        #     if len(ego_by_cav_list) == 1:
        #         pred_box3d_tensor.append(cav_preds[item])
        #         scores.append(cav_probs[item])
        #     elif len(ego_by_cav_list) == 2:
        #         max_prob = 0
        #         max_indx = 0
        #         for sub_indexx in range(len(ego_by_cav_list)):
        #             if cav_probs[ego_by_cav_list[sub_indexx]] > max_prob:
        #                 max_prob = cav_probs[ego_by_cav_list[sub_indexx]]
        #                 max_indx = ego_by_cav_list[sub_indexx]
        #         pred_box3d_tensor.append(cav_preds[max_indx])
        #         scores.append(cav_probs[max_indx])
        #     #if i == 0:
        #     ego_index = range(len(ego_preds))
        #     for item in ego_index:
        #         if item not in ego_visited:
        #             v1 = ego_preds[item]
        #             v1 = v1[np.newaxis,:,:]
                    
        #             centers = v1.mean(axis=1)[:,:2]
        #             #vehicle_point = torch.stack(vehicle_location)[:,:,:2]
        #             vehicle_point = torch.stack(vehicle_location).squeeze(1)[:,:2]
        #             distance = torch.sqrt(torch.sum((centers - vehicle_point) ** 2, axis=1))

        #             flag = True
        #             cur_dis = distance[0]
        #             for sub_idx_dis in range(len(distance)):
        #                 sub_dis = distance[sub_idx_dis]
        #                 if cur_dis > sub_dis:
        #                     flag = False
        #                 if sub_dis < 1:
        #                     flag = True
        #                     break
        #             if flag:
        #                 pred_box3d_tensor.append(ego_preds[item])
        #                 scores.append(ego_probs[item])


        # if len(pred_box3d_tensor) != 0:
        #     pred_box3d_tensor = torch.stack(pred_box3d_tensor)
        #     scores = torch.stack(scores)
        # if len(total_v1) == 1:
        #     pred_box3d_tensor = total_v1[0]
        #     scores = total_v2[0]

        # # pred_box3d_tensor = torch.vstack(total_v1)
        # # scores = torch.cat(total_v2)
        # keep_index = box_utils.nms_rotated(pred_box3d_tensor,
        #                                    scores,
        #                                    self.params['nms_thresh']
        #                                    )
        # pred_box3d_tensor = pred_box3d_tensor[keep_index]
        # scores = scores[keep_index]
        # mask = \
        #     box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        # pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        # scores = scores[mask]



        ##separate  and fusionicra



        ####separate  and fusion icra
        # total_v1 = []
        # total_v2 = []
        # for v1,v2 in zip(pred_box2d_list,pred_box3d_list):   
        #     probs = v1[:, -1].cpu()
        #     preds = v2.cpu()
            
        #     iou_mat = compute_self_iou_mat(preds)
        #     clusters, visited, selected = [], [], []
        #     for idx, ious in enumerate(iou_mat):
        #         if idx in visited:
        #             continue
        #         neighbor_idxs = np.nonzero(ious)[0]
        #         clusters.append(neighbor_idxs)
        #         visited.extend(neighbor_idxs)
        #     for cluster in clusters:
        #         sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
        #         sub_probs = probs[cluster]
        #         values = sub_iou_mat.dot(sub_probs)
        #         soft_bools = softmax(values / 1e-6)
        #         bools = soft_bools > 0.5
        #         selected.extend(cluster[bools])
        #     pred_box3d_tensor = preds[selected]
        #     scores = probs[selected]

        #     total_v1.append(pred_box3d_tensor)
        #     total_v2.append(scores)

        # pred_box3d_tensor = torch.vstack(total_v1)
        # scores = torch.cat(total_v2)
        # preds = pred_box3d_tensor.cpu()
        # probs = scores.cpu()
        # iou_mat = compute_self_iou_mat(preds)
        # clusters, visited, selected = [], [], []
        # for idx, ious in enumerate(iou_mat):
        #     if idx in visited:
        #         continue
        #     neighbor_idxs = np.nonzero(ious)[0]
        #     clusters.append(neighbor_idxs)
        #     visited.extend(neighbor_idxs)
        # for cluster in clusters:
        #     sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
        #     sub_probs = probs[cluster]
        #     values = sub_iou_mat.dot(sub_probs)
        #     soft_bools = softmax(values / 1e-6)
        #     bools = soft_bools > 0.5
        #     selected.extend(cluster[bools])
        # pred_box3d_tensor = preds[selected]
        # scores = probs[selected]
        ####separate  and fusionicra


        ##separate  and fusion nms
        # total_v1 = []
        # total_v2 = []
        # for v1,v2 in zip(pred_box2d_list,pred_box3d_list):   
        #     scores = v1[:, -1]
        #     pred_box3d_tensor = v2
        #     keep_index = box_utils.nms_rotated(pred_box3d_tensor,
        #                                     scores,
        #                                     self.params['nms_thresh']
        #                                     )
        #     pred_box3d_tensor = pred_box3d_tensor[keep_index]
        #     scores = scores[keep_index]
        #     mask = \
        #         box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        #     pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        #     total_v1.append(pred_box3d_tensor)
        #     scores = scores[mask]
        #     total_v2.append(scores)
        # pred_box3d_tensor = torch.vstack(total_v1)
        # scores = torch.cat(total_v2)
        # keep_index = box_utils.nms_rotated(pred_box3d_tensor,
        #                                    scores,
        #                                    self.params['nms_thresh']
        #                                    )
        # pred_box3d_tensor = pred_box3d_tensor[keep_index]
        # scores = scores[keep_index]
        # mask = \
        #     box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        # pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        # scores = scores[mask]
        #separate  and fusion nms





        # shape: (N, 5)
        # pred_box2d_list = torch.vstack(pred_box2d_list)
        # # scores
        # scores = pred_box2d_list[:, -1]
        # # predicted 3d bbx
        # pred_box3d_tensor = torch.vstack(pred_box3d_list)
        

        #cluster
        # dbscan = DBSCAN(eps=0.7, min_samples=2, metric='precomputed')
        # preds = pred_box3d_tensor.cpu()
        # probs = scores.cpu()
        # iou_mat = compute_self_iou_mat(preds)
        # labels = dbscan.fit_predict(iou_mat)
        # clusters, visited, selected = [], [], []
        # for idx, ious in enumerate(iou_mat):
        #     if idx in visited:
        #         continue
        #     neighbor_idxs = np.nonzero(ious)[0]
        #     clusters.append(neighbor_idxs)
        #     visited.extend(neighbor_idxs)
        # for cluster in clusters:
        #     sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
        #     sub_probs = probs[cluster]
        #     values = sub_iou_mat.dot(sub_probs)
        #     soft_bools = softmax(values / 1e-6)
        #     bools = soft_bools > 0.5
        #     selected.extend(cluster[bools])
        # pred_box3d_tensor = preds[selected]
        # scores = probs[selected]

        #icra
        # pred_box2d_list = torch.vstack(pred_box2d_list)
        # # scores
        # scores = pred_box2d_list[:, -1]
        # # predicted 3d bbx
        # pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # preds = pred_box3d_tensor.cpu()
        # probs = scores.cpu()
        # iou_mat = compute_self_iou_mat(preds)
        # clusters, visited, selected = [], [], []
        # for idx, ious in enumerate(iou_mat):
        #     if idx in visited:
        #         continue
        #     neighbor_idxs = np.nonzero(ious)[0]
        #     clusters.append(neighbor_idxs)
        #     visited.extend(neighbor_idxs)
        # for cluster in clusters:
        #     sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
        #     sub_probs = probs[cluster]
        #     values = sub_iou_mat.dot(sub_probs)
        #     soft_bools = softmax(values / 1e-6)
        #     bools = soft_bools > 0.5
        #     selected.extend(cluster[bools])
        # pred_box3d_tensor = preds[selected]
        # scores = probs[selected]
        
        ##nms
        # pred_box2d_list = torch.vstack(pred_box2d_list)
        # # scores
        # scores = pred_box2d_list[:, -1]
        # # predicted 3d bbx
        # pred_box3d_tensor = torch.vstack(pred_box3d_list)

        # scores = scores[keep_index]


        # keep_index = box_utils.nms_rotated(pred_box3d_tensor,
        #                                    scores,
        #                                    self.params['nms_thresh']
        #                                    )

        # pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # # select cooresponding score
        # scores = scores[keep_index]

        # # filter out the prediction out of the range.
        # mask = \
        #     box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        # pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        # scores = scores[mask]
        


        # total_v1 = []
        # total_v2 = []
        # for v1,v2 in zip(pred_box2d_list,pred_box3d_list):   
        #     scores = v1[:, -1]
        #     pred_box3d_tensor = v2
        #     keep_index = box_utils.nms_rotated(pred_box3d_tensor,
        #                                     scores,
        #                                     self.params['nms_thresh']
        #                                     )
        #     pred_box3d_tensor = pred_box3d_tensor[keep_index]
        #     scores = scores[keep_index]
        #     mask = \
        #         box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        #     pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        #     total_v1.append(pred_box3d_tensor)
        #     scores = scores[mask]
        #     total_v2.append(scores)
        #     #break

        # for i in range(len(total_v1)):
        #     preds = total_v1[i]
        #     centers = preds.mean(axis=1)[:,:2].cpu()
        #     probs = total_v2[i]
        #     vehicle_point = vehicle_location[i][:,:2]
        #     distance = torch.sqrt(torch.sum((centers - vehicle_point) ** 2, axis=1))
        #     for indx in range(len(distance)):
        #         if distance[indx] > 20:
        #             total_v2[i][indx] = total_v2[i][indx] * 0.7
        #     indx_list = torch.argsort(distance)

        # pred_box3d_tensor = torch.vstack(total_v1)
        # scores = torch.cat(total_v2)
        # keep_index,single_index = box_utils.v2v_nms_rotated(pred_box3d_tensor,
        #                                    scores,
        #                                    self.params['nms_thresh']
        #                                    )

        
        # index = 0
        # for i in range(len(total_v1)):
        #     Is_vehicle_overlap = []
        #     v1 = total_v1[i]
        #     for j in range(len(total_v1[i])):
        #         if index in single_index:
        #             cur_box = total_v1[i][j].cpu()
        #             cur_box = cur_box[np.newaxis,:,:]
        #             cur_box_center = cur_box.mean(axis=1)[:,:2]
        #             vehicle_point = torch.stack(vehicle_location).squeeze(1)[:,:2]
        #             distance = torch.sqrt(torch.sum((cur_box_center - vehicle_point) ** 2, axis=1))
                    
        #             flag = True
        #             cur_dis = distance[i]
        #             for sub_idx_dis in range(len(distance)):
        #                 sub_dis = distance[sub_idx_dis]
        #                 if cur_dis - 30 > sub_dis:
        #                     flag = False
        #                 # if sub_dis < 1:
        #                 #     flag = True
        #                 #     break
        #             if not flag:
        #                 keep_index = np.delete(keep_index, np.where(keep_index == index))
        #         index += 1


        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)

        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
