# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from dipy.core.geometry import cart2sphere, sphere2cart
# sys.path.insert(0, "../tools/fat_dataset")
from convert_fat_coco import *

class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=800,
        categories=None,
        topk_inplane_rotations=9,
        topk_viewpoints=9,
        # viewpoints_xyz=None,
        # inplane_rotations=None,
        # fixed_transforms_dict=None,
        # camera_intrinsics=None
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        self.CATEGORIES += categories
        self.topk_inplane_rotations = topk_inplane_rotations
        self.topk_viewpoints = topk_viewpoints
        # self.viewpoints_xyz = viewpoints_xyz
        # self.inplane_rotations = inplane_rotations
        # self.fixed_transforms_dict = fixed_transforms_dict
        # self.camera_intrinsics = camera_intrinsics

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        print("Using model at : {}".format(cfg.MODEL.WEIGHT))
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image, use_thresh=False):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result, centroids, boxes = self.overlay_boxes(result, top_predictions)
        
        if self.cfg.MODEL.MASK_ON:
            mask_list, overall_binary_mask = self.get_all_masks(result, top_predictions)
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)
        if self.cfg.MODEL.POSE_ON:
            # img_list = self.render_poses(top_predictions)
            rotation_list = self.get_all_rotations(top_predictions, use_thresh)

        if self.cfg.MODEL.MASK_ON:
            if self.cfg.MODEL.POSE_ON:
                return result, mask_list, rotation_list, centroids, boxes, overall_binary_mask
            return result, mask_list, centroids, boxes, overall_binary_mask
        else:
            return result, centroids

    def get_all_rotations(self, top_predictions, use_thresh):
        top_viewpoint_ids, top_inplane_rotation_ids = \
            self.select_top_rotations(top_predictions, use_thresh=use_thresh)
        labels = top_predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        rotations = {}
        rotations['top_viewpoint_ids'] = top_viewpoint_ids
        rotations['top_inplane_rotation_ids'] = top_inplane_rotation_ids
        rotations['labels'] = labels

        return rotations

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]
        print("Found : {}".format(predictions))
        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        # if prediction.has_field("viewpoint_scores") and prediction.has_field("inplane_rotation_scores"):
        #     prediction = self.select_top_rotations(prediction)
            
        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_rotations(self, prediction, use_thresh=False):
        top_viewpoint_ids = []
        top_inplane_rotation_ids = []

        viewpoint_scores = prediction.get_field("viewpoint_scores")
        inplane_rotation_scores = prediction.get_field("inplane_rotation_scores")
        if use_thresh:
            print("Using score threshold for viewpoints and inplane rotation")
            top_viewpoint_ids = torch.topk(viewpoint_scores, self.topk_viewpoints, dim=1, largest=True, sorted=True)[1]
            top_inplane_rotation_ids = torch.topk(inplane_rotation_scores, self.topk_inplane_rotations, dim=1)[1]
            print(top_viewpoint_ids)
            # top_viewpoint_scores = viewpoint_scores > 0.01
            # top_inplane_rotation_scores = inplane_rotation_scores > 0.1
            # # Do for each box
            # for i in range(top_viewpoint_scores.shape[0]):
            #     # Add indexes which are non-zero after thresholding
                
            #     top_viewpoint_ids.append(
            #         torch.nonzero(top_viewpoint_scores[i, :]).numpy().flatten().tolist()
            #     )
            #     top_inplane_rotation_ids.append(
            #         torch.nonzero(top_inplane_rotation_scores[i, :]).numpy().flatten().tolist()
            #     )
        else:
            top_viewpoint_ids = torch.argmax(viewpoint_scores, dim=1).numpy().tolist()
            top_inplane_rotation_ids = torch.argmax(inplane_rotation_scores, dim=1).numpy().tolist()
        
        # prediction.add_field("viewpoint_ids", top_viewpoint_ids)
        # prediction.add_field("inplane_rotation_ids", top_inplane_rotation_ids)
       
        return top_viewpoint_ids, top_inplane_rotation_ids
        # return prediction

    # def render_poses(self, top_predictions):
    #     top_viewpoint_ids, top_inplane_rotation_ids = \
    #         self.select_top_rotations(top_predictions, use_thresh=False)
    #     labels = top_predictions.get_field("labels").tolist()
    #     labels = [self.CATEGORIES[i] for i in labels]

    #     print("Predicted top_viewpoint_ids : {}".format(top_viewpoint_ids))
    #     print("Predicted top_inplane_rotation_ids : {}".format(top_inplane_rotation_ids))
    #     print(labels)

    #     img_list = []

    #     for i in range(len(top_viewpoint_ids)):
    #         viewpoint_id = top_viewpoint_ids[i]
    #         inplane_rotation_id = top_inplane_rotation_ids[i]
    #         label = labels[i]
    #         fixed_transform = self.fixed_transforms_dict[label]
    #         theta, phi = get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
    #         inplane_rotation_angle = get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
    #         xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
    #         print("Recovered rotation : {}".format(xyz_rotation_angles))
    #         rgb_gl, depth_gl = render_pose(label, fixed_transform, self.camera_intrinsics, xyz_rotation_angles, [0,0,100])
    #         img_list.append([rgb_gl, depth_gl])

    #     return img_list
    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)

        # Rough way to remove double detections (Aditya)
        # Keep each label with highest score
        sorted_predictions = predictions[idx]
        labels = sorted_predictions.get_field("labels")
        labels = [self.CATEGORIES[i] for i in labels]
        labels_found = []
        filtered_idx = []
        for li in range(len(labels)):
            label = labels[li]
            if label not in labels_found:
                labels_found.append(label)
                filtered_idx.append(li)
            else:
                continue
        # print(labels)
        # print(labels_found)
        # filtered_idx = torch.tensor(filtered_idx)
        # print(filtered_idx)
        return sorted_predictions[filtered_idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()
        centroids = []
        box_list = []
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
            box_list.append([top_left, bottom_right])
            centroids.append((np.array(bottom_right) + np.array(top_left))/2)


        return image, centroids, box_list

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()
        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite
    
    def get_all_masks(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()
        mask_list = []
        overall_binary_mask = np.zeros((image.shape[0], image.shape[1]))

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            img = np.zeros((image.shape[0], image.shape[1]))
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            mask_image = cv2.fillPoly(img, pts = contours, color=(255))
            overall_binary_mask = cv2.fillPoly(overall_binary_mask, pts = contours, color=(255))
            mask_list.append(mask_image)
        return mask_list, overall_binary_mask

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1
            )

        return image

import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
