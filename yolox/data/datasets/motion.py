from loguru import logger

import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.ops import box_convert
from gluoncv.torch.data.gluoncv_motion_dataset.dataset import (
    GluonCVMotionDataset,
    AnnoEntity
)

import os
import random
from PIL import Image

from .datasets_wrapper import Dataset

from yolox.utils import is_main_process


class MotionDataset(Dataset):
    """
    GluonCV Motion dataset class.
    """

    def __init__(self,
                 root_path=None,
                 anno_file=None,
                 split_file=None,
                 filter_fn=None,
                 sampling_interval=1000,
                 clip_len=1000,
                 tsm=False,
                 is_train=True,
                 img_size=(736, 1280),
                 amodal=False,
                 preproc=None,
                 debug_limit=None,
                 save_image_examples=False,
                 ):
        super().__init__(img_size)

        dataset_name = root_path.split('datasets/')[-1].split('/motion')[0]
        logger.info(f'Dataset name: {dataset_name}')

        self.tsm = tsm
        if self.tsm:
            logger.warning('TSM is enabled!')
        
        frames_in_clip = 2 if self.tsm else 1

        assert is_train is True, "This dataset clss only supports training"
        assert (2 >= frames_in_clip > 0), "frames_in_clip has to be 1 or 2"

        self.motion_dataset = GluonCVMotionDataset(anno_file,
                                                   root_path=root_path,
                                                   split_file=split_file)
        logger.info(f'GluonCVMotionDataset description: {self.motion_dataset.description}')

        self.data = dict(self.motion_dataset.train_samples)
        self.debug_limit = debug_limit
        if self.debug_limit:
            self.debug_limit = int(self.debug_limit)
            logger.info(f'Debug limit: {self.debug_limit}')
            self.data = dict(random.sample(self.data.items(), self.debug_limit))
        self.img_size = img_size
        self.clip_len = clip_len
        self.filter_fn = filter_fn
        self.frames_in_clip = min(clip_len, frames_in_clip)
        logger.info(f'MotionDataset frames_in_clip: {self.frames_in_clip}')
        logger.info(f'MotionDataset sampling_interval: {sampling_interval}')
        logger.info(f'MotionDataset clip_len: {clip_len}')

        # Write/get clips list
        clips_file = f'{dataset_name}_clips_list_full_sampling_interval_{sampling_interval}.pkl'
        if os.path.exists(clips_file):
            logger.warning(f'Loading clips from {clips_file}')
            with open(clips_file, 'rb') as f:
                self.clips = pickle.load(f)
        else:
            # Process dataset to get all valid video clips
            self.clips = self.get_video_clips(sampling_interval_ms=sampling_interval)
            if debug_limit is None and is_main_process():
                with open(clips_file, 'wb') as f:
                    pickle.dump(self.clips, f)

        logger.info(f'Number of clips per epoch: {len(self.clips)}')

        self.amodal = False
        self.preproc = preproc
        self.annotations = dict()
        self.save_image_examples = save_image_examples
        if self.preproc:
            self.means = self.preproc.means
            self.stds = self.preproc.std
            self.inv_stds = [1/std for std in self.stds]
            self.neg_means = [-mean for mean in self.means]
            self.inverse_transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                              std = self.inv_stds),
                                                         transforms.Normalize(mean = self.neg_means,
                                                                              std = [ 1., 1., 1. ]),
                                                        ])

    def pull_item(self, item_id):
        video, target = [], []
        (sample_id, clip_frame_ids) = self.clips[item_id]
        video_info = self.data[sample_id]
        video_reader = video_info.get_data_reader()
        # Randomly sampling self.frames_in_clip frames
        # And keep their relative temporal order
        rand_idxs = sorted(random.sample(clip_frame_ids, self.frames_in_clip))
        im_types = [type(video_reader[frame_idx][0]) for frame_idx in rand_idxs]

        for frame_idx in rand_idxs:
            im = video_reader[frame_idx][0]

            # Error checking
            tries = 0
            while im is None:
                tries += 1
                logger.warning(f'===== item_id {item_id} im is None: resampling: tries={tries} =====')
                frame_idx = sorted(random.sample(clip_frame_ids, self.frames_in_clip))[-1]
                im = video_reader[frame_idx][0]

            entities = video_info.get_entities_for_frame_num(frame_idx)
            if self.filter_fn is not None:
                entities, _ = self.filter_fn(entities, meta_data=video_info.metadata)

            boxes = self.entity2target(im, entities)
            video.append(im)
            target.append(boxes)

#         # Video clip-level augmentation # TODO: remove?
#         if self.transforms is not None:
#             video, target = self.transforms(video, target)

        video_info.clear_lazy_loaded()

        def _process_img_boxlist(img, boxlist):
            height = img.shape[0]
            width = img.shape[1]
            num_boxes = boxlist.bbox.shape[0]
            num_labels = len(boxlist.get_field('labels'))
            try:
                res = np.zeros((num_boxes, 6))
                res[:,:4] = boxlist.bbox
                res[:,4] = boxlist.get_field('labels')
                res[:,4] -= 1
                res[:,5] = boxlist.get_field('ids')
                target = res
                self.annotations[item_id] = target
                # TODO: last index should be file_name
                img_info = (height, width, frame_idx, sample_id, 'blah.jpeg')
            except Exception as e:
                logger.error(f'num_boxes: {num_boxes}, num_labels: {num_labels}')
                logger.exception(f'Failed building res in pull_item! seq={sample_id}')
                raise e
            return img, target, img_info, sample_id

        # Standard single image detection
        if len(video) == 1:
            img = video[0]  # 1 of 1 image

            if not isinstance(img, np.ndarray):
                logger.warning('img is not np.ndarray (PIL?)')

            # Get annotations
            boxlist = target[0]  # 1 of 1 annotations for image
            img, target, img_info, sample_id = _process_img_boxlist(img, boxlist)
            return img, target, img_info, sample_id

        # Pair of images
        elif len(video) == 2:
            imgs, targets, img_infos, sample_ids = [],[],[],[]
            for i, b in zip(video, target):
                img, target, img_info, sample_id = _process_img_boxlist(i, b)
                imgs.append(img)
                targets.append(target)
                img_infos.append(img_info)
                sample_ids.append(sample_id)
            return imgs, targets, img_infos, sample_ids

    def load_anno(self, index):
        if index not in self.annotations:
            img, target, img_id, img_info = self.pull_item(index)
            self.annotations[index] = target
        return self.annotations[index]

    @Dataset.resize_getitem
    def __getitem__(self, index):
        imgs, targets, img_infos, img_ids = self.pull_item(index)

        # list of imgs, targets
        if isinstance(imgs, list):
            imgs_ret, targets_ret = [],[]
            for img, target in zip(imgs, targets):
                if self.preproc is not None:
                    img, target = self.preproc(img, target, self.input_dim)
                imgs_ret.append(img)
                targets_ret.append(target)
            ret_val = imgs_ret, targets_ret, img_infos, img_ids    

        # Single img, target pair
        else:
            img, target, img_info, img_id = imgs, targets, img_infos, img_ids
            if self.preproc is not None:
                img, target = self.preproc(img, target, self.input_dim)
            ret_val = img, target, img_info, img_id

            # TODO, make save_image work for multiframe
            if self.save_image_examples:
                self.save_image(index, ret_val)

        return ret_val

    def save_image(self, index, ret_val):

        # Save images
        img = ret_val[0]
        target = ret_val[1]
        img_info = ret_val[2]

        # mosaic
        if len(img_info) == 2:
            seq = f'mosaic_index_{index}'
            frame = 'mosaic_multiframe'
        else:
            img_id = ret_val[3]
            seq = img_id
            frame = img_info[2]

        def draw_bboxes(img, target):
            annotated_img = img.copy()
            annotated_img = torch.tensor(annotated_img)

            target_copy = target.copy()
            target_copy = torch.tensor(target_copy)

            if self.preproc:
                annotated_img = self.inverse_transform(annotated_img)
                annotated_img *= 255
                annotated_img = np.array(np.moveaxis(annotated_img.cpu().numpy(), 0, -1))
                target_copy[:, 1:5] = box_convert(target_copy[:, 1:5], in_fmt='cxcywh',out_fmt='xyxy')
                label_idx = 0
                x1_idx, y1_idx, x2_idx, y2_idx = 1, 2, 3, 4
            else:
                annotated_img = np.array(annotated_img.cpu().numpy())
                label_idx = 4
                x1_idx, y1_idx, x2_idx, y2_idx = 0, 1, 2, 3

            annotated_img = annotated_img.astype(np.uint8)
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            for n, res in enumerate(target_copy):
                if torch.sum(res) == 0:
                    continue
                x1 = int(res[x1_idx])
                y1 = int(res[y1_idx])
                x2 = int(res[x2_idx])
                y2 = int(res[y2_idx])
                label = str(int(res[label_idx]))
                targetid = int(res[5])
                imgHeight, imgWidth, _ = annotated_img.shape
                thick = 1
                cv2.rectangle(annotated_img,(x1, y1), (x2, y2), (0,0,255), thick)
                cv2.putText(annotated_img, f'{label}_{targetid}', (x1, y1 - 12), 
                            0, 1e-3 * imgHeight, (0,255,0), thick//3)
            return annotated_img
        annotated_img = draw_bboxes(img, target)
        h,w,c = annotated_img.shape
        with_preproc= 'withPreproc' if self.preproc else 'noPreproc'
        save_path = os.path.join('./image_samples', f'MotionDataset_{with_preproc}_{str(int(index)).zfill(7)}_{seq}_{frame}_{str(h)}_{str(w)}.jpeg')
        cv2.imwrite(save_path, annotated_img)
        print(f'save_image MotionDataset: {save_path}')

    def __len__(self):
        return len(self.clips)

    def get_video_clips(self, sampling_interval_ms=1000):
        """
        Process the long videos to a small video chunk (with self.clip_len seconds)
        Video clips are generated in a temporal sliding window fashion
        """
        video_clips = []
        # weight_info = []
        logger.info('Getting video clips (clips per epoch)...')
        progress_bar = tqdm if is_main_process() else iter
        for (sample_id, sample) in progress_bar(self.data.items()):
            frame_idxs_with_anno = sample.get_non_empty_frames(self.filter_fn)
            len_id_entity_dict = len(sample.id_entity_dict)
            if len(frame_idxs_with_anno) == 0:
                continue
            # The video clip may not be temporally continuous
            start_frame = min(frame_idxs_with_anno)
            end_frame = max(frame_idxs_with_anno)
            # make sure that the video clip has at least two frames
            clip_len_in_frames = max(self.frames_in_clip, int(self.clip_len / 1000. * sample.fps))
            sampling_interval = int(sampling_interval_ms / 1000. * sample.fps)
            # print(f'sampling_interval={sampling_interval}, clip_len_in_frames='
            #       f'{clip_len_in_frames}, start_frame={start_frame}, '
            #       f'end_frame={end_frame}, fps={sample.fps}, clip_len='
            #       f'{self.clip_len}, sampling_interval={sampling_interval}, '
            #       f'frame_idxs_with_anno={frame_idxs_with_anno}')
            for n, idx in enumerate(range(start_frame, end_frame,
                                          sampling_interval)):
                # print(f'\tsample_id: {sample_id}, n={n}, idx={idx}, '
                #       f'idx+clip_len_in_frames: {idx+clip_len_in_frames}')
                clip_frame_ids = []
                # only include frames with annotation within the video clip
                for frame_idx in range(idx, idx + clip_len_in_frames):
                    if frame_idx in frame_idxs_with_anno:
                        clip_frame_ids.append(frame_idx)
                # Only include video clips that have at least self.frames_in_clip annotating frames
                if len(clip_frame_ids) >= self.frames_in_clip:
                    video_clips.append((sample_id, clip_frame_ids))
#                     weight_info.append({
#                         'sample_id': sample_id,
#                         'len_id_entity_dict': len_id_entity_dict,
#                     })
#                     print(f'{sample_id}, len_id_entity_dict: {len_id_entity_dict}')
                sample.clear_lazy_loaded()

        # self.weight_info = weight_info
        return video_clips

    def entity2target(self, im: Image, entities: [AnnoEntity]):
        """
        Wrap up the entity to maskrcnn-benchmark compatible format - BoxList
        """
        boxes = [entity.bbox for entity in entities]
        ids = [int(entity.id) for entity in entities]
        # we only consider person tracking for now,
        # thus all the labels are 1,
        # reserve category 0 for background during training
        # int_labels = [1 for _ in entities]
        int_labels = [list(entity.labels.values())[0] for entity in entities]

        boxes = torch.as_tensor(boxes).reshape(-1, 4)

        # Check if opencv array, default to PIL
        if isinstance(im, np.ndarray):
            # height, width, channels = img.shape
            size = (im.shape[1], im.shape[0])
        else:
            # (width, height) = im.size
            size = im.size

        boxes = BoxList(boxes, size, mode='xywh').convert('xyxy')
        if not self.amodal:
            # boxes = boxes.clip_to_image(remove_empty=False)
            boxes, keep = boxes.clip_to_image(remove_empty=True)
        boxes.add_field('labels', torch.as_tensor(int_labels, 
                                                  dtype=torch.int64)[keep])
        boxes.add_field('ids', torch.as_tensor(ids, dtype=torch.int64)[keep])
        return boxes


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box
        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1

        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep], keep
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s