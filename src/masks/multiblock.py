# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.input_size = input_size
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement
    
    def _sample_block_mask_superpoints(self, b_size, count, acceptable_regions=None):
        h, w = b_size
        
        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        
        count_vec = count.view(-1)
        top_10 = torch.topk(count_vec,10)
        top_indices = torch.unravel_index(top_10[1],count.shape)
        top_indices = torch.stack(list(top_indices), dim=1)
        
        while not valid_mask:            
            # -- Sample block top-left corner
            rand_i = torch.randint(0, 10, (1,))[0]
            top = top_indices[rand_i][0]
            left = top_indices[rand_i][1]
            
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement
    
    def _count_matrix_from_keypoints(self, img_path, patch_size):
        superpoint_path = '/dsi/scratch/home/dsi/rinaVeler/datasets/superpoint/'
        file_name = img_path.split(os.sep)[-3]+'_'+img_path.split(os.sep)[-2]+'_'+os.path.splitext(os.path.basename(img_path))[0]
        pose_name = superpoint_path + file_name + '.npy'
        pose = np.load(pose_name, allow_pickle=True)
        keypoints = pose.item().get('keypoints')[0].cpu().numpy().astype(int)
        keypoints_mat = torch.zeros(self.input_size)
        for ind in keypoints: keypoints_mat[ind[0],ind[1]] = 1
        patch_kp = torch.zeros(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                patch_kp[i,j] = torch.sum(keypoints_mat[i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size])
        
        count = torch.zeros([self.height-patch_size[0]+1, self.width-patch_size[1]+1])
        for i in range(0,count.shape[0]):
            for j in range(0,count.shape[1]):
                count[i,j] =0 
                for x in range(i,i+patch_size[0]):
                    for y in range(j,j+patch_size[1]):
                        count[i,j] += patch_kp[x,y]
        return count
    
    def _count_matrix_from_canny(self, img_path, patch_size):
        canny_path = '/dsi/scratch/home/dsi/rinaVeler/datasets/canny/'
        file_name = img_path.split(os.sep)[-3]+'_'+img_path.split(os.sep)[-2]+'_'+os.path.splitext(os.path.basename(img_path))[0]
        canny_name = canny_path + file_name + '.npy'
        canny = torch.from_numpy(np.load(canny_name))
        patch_edg = torch.zeros(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                patch_edg[i,j] = torch.sum(canny[i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size])
        
        count = torch.zeros([self.height-patch_size[0]+1, self.width-patch_size[1]+1])
        for i in range(0,count.shape[0]):
            for j in range(0,count.shape[1]):
                count[i,j] =0 
                for x in range(i,i+patch_size[0]):
                    for y in range(j,j+patch_size[1]):
                        count[i,j] += patch_edg[x,y]
        return count
        
    def _build_cdf_from_keypoints(self, img_path):
        superpoint_path = '/dsi/scratch/home/dsi/rinaVeler/datasets/superpoint/'
        file_name = img_path.split(os.sep)[-3]+'_'+img_path.split(os.sep)[-2]+'_'+os.path.splitext(os.path.basename(img_path))[0]
        print(file_name)
        pose_name = superpoint_path + file_name + '.npy'
        pose = np.load(pose_name, allow_pickle=True)
        keypoints = pose.item().get('keypoints')[0]
        scores = pose.item().get('scores')[0]
        k_x = keypoints[:,0].cpu().numpy()
        k_y = keypoints[:,1].cpu().numpy()
        hist, x_bins, y_bins = np.histogram2d(k_x, k_y, bins=(16,16))
        x_bin_midpoints = x_bins[:-1] + np.diff(x_bins)/2
        y_bin_midpoints = y_bins[:-1] + np.diff(y_bins)/2
        cdf = np.cumsum(hist.ravel())
        cdf = cdf / cdf[-1]
        
        return cdf, x_bin_midpoints, y_bin_midpoints

    def _random_from_cdf(self, cdf, x_bin_midpoints, y_bin_midpoints, n):        
        values = np.random.rand(n)
        value_bins = np.searchsorted(cdf, values)
        x_idx, y_idx = np.unravel_index(value_bins,
                                        (len(x_bin_midpoints),
                                        len(y_bin_midpoints)))
        random_from_cdf = np.column_stack((x_bin_midpoints[x_idx],
                                        y_bin_midpoints[y_idx]))
        new_x, new_y = random_from_cdf.T
        return (new_x//16).astype(int), (new_y//16).astype(int)

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        
        for b in range(B): 
                
            n_kp_psize = self._count_matrix_from_keypoints(batch[b][2], p_size)
            n_edg_psize = self._count_matrix_from_canny(batch[b][2], p_size)
            n_kp_esize = self._count_matrix_from_keypoints(batch[b][2], e_size)
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                case = torch.randint(0,2,(1,)) # To choosse only one method choose 1 or 0
                if case == 1:
                    mask, mask_C = self._sample_block_mask_superpoints(p_size, n_kp_psize)
                else:
                    mask, mask_C = self._sample_block_mask_superpoints(p_size, n_edg_psize)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                #mask, _ = self._sample_block_mask_superpoints(e_size, n_kp_esize,acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
