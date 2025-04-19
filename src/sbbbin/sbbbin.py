# Copyright 2025 Janik Haitz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file includes code from the sbb_binarization project,
# available at https://github.com/qurator-spk/sbb_binarization and licensed under
# Apache 2.0 license https://github.com/qurator-spk/sbb_binarization/LICENSE.

from pathlib import Path
from typing import Any
from os import environ, devnull
import sys
import logging

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(devnull, 'w')
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models, backend
sys.stderr = stderr
cv2.setLogLevel(2)
tf.get_logger().setLevel(logging.ERROR)

logger = logging.getLogger("sbbbin")

PROJECTION_DIM = 64
PATCH_SIZE = 1
NUM_PATCHES = 196  # 14x14


def resize_image(image, height, width):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


class Patches(layers.Layer):
    def __init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = PATCH_SIZE
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size,})
        return config
    

class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = NUM_PATCHES
        self.projection = layers.Dense(units=PROJECTION_DIM)
        self.position_embedding = layers.Embedding(input_dim=NUM_PATCHES, 
                                                   output_dim=PROJECTION_DIM)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_patches': self.num_patches,
                       'projection': self.projection,
                       'position_embedding': self.position_embedding,})
        return config
    

class SbbBinarizer:
    def __init__(self, models_path: Path, force_cpu: bool = False):
        """
        Args:
            models_path: Path to the directory containing model directories.
        """
        self.start_session(force_cpu)
        self.model_dirs = list([fp for fp in models_path.glob('*') if fp.is_dir()])
        self.loaded_models = []
        for fp in self.model_dirs:
            logger.debug(f"Loading model from {fp}")
            self.loaded_models.append(self._load_model(fp))
        
    def run(self, image: cv2.typing.MatLike, use_patches: bool = False) -> cv2.typing.MatLike:
        """
        Predict the binarized image for an input image.
        Args:
            image: Input cv2 image object.
            use_patches: If set to True, use patches (better results). Defaults to False.
        Returns:
            A binarized cv2 image object.
        """
        img_last = 0
        for n, (model, fp) in enumerate(zip(self.loaded_models, self.model_dirs)):
            logger.debug(f"Predicting with model {fp} [{n+1}/{len(self.model_dirs)}]")
            res = self._predict(image, model, use_patches)
            img_fin = np.zeros((res.shape[0], res.shape[1], 3))
            res[:, :][res[:, :] == 0] = 2
            res = res - 1
            res = res * 255
            img_fin[:, :, 0] = res
            img_fin[:, :, 1] = res
            img_fin[:, :, 2] = res
            img_fin = img_fin.astype(np.uint8)
            img_fin = (res[:, :] == 0) * 255
            img_last = img_last + img_fin
        img_last[:, :][img_last[:, :] > 0] = 255
        return(img_last[:, :] == 0) * 255
            
    def start_session(self, force_cpu: bool = False):
        if force_cpu:
            tf.config.set_visible_devices([], "GPU")
            logger.info("Force CPU computation")
        else:
            if not tf.config.list_physical_devices("GPU"):
                logger.warning("No GPU found. Please ensure that a compatible GPU is available.")
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)
        backend.set_session(self.session)
        
    def end_session(self):
        backend.clear_session()
        self.session.close()
        del self.session
        
    def _load_model(self, fp: Path) -> tuple[Any, int, int]:
        """
        Loads a tensorflow model and returns the loaded model with additional information
        Args:
            fp: Path to model
        Returns:
            A Tuple containing the loaded model, the model height and model width.
        """
        try:
            model = models.load_model(fp, compile=False)
            self.margin_percent = 0.1
        except Exception:
            model = models.load_model(fp, compile=False, custom_objects={"PatchEncoder": PatchEncoder, "Patches": Patches})
            self.margin_percent = 0.15
        model_height = model.layers[len(model.layers)-1].output_shape[1]
        model_width = model.layers[len(model.layers)-1].output_shape[2]
        return model, model_height, model_width
    
    def _predict(self, img: cv2.typing.MatLike, model: tuple[Any, int, int], use_patches: bool = False):
        backend.set_session(self.session)
        model, model_height, model_width = model
        
        img_org_h = img.shape[0]
        img_org_w = img.shape[1]
        
        if img.shape[0] < model_height and img.shape[1] >= model_width:
            img_padded = np.zeros((model_height, img.shape[1], img.shape[2]))
            index_start_h = int(abs(img.shape[0] - model_height) / 2.0)
            index_start_w = 0
            img_padded [index_start_h:index_start_h + img.shape[0], :, :] = img[:, :, :]        
        elif img.shape[0] >= model_height and img.shape[1] < model_width:
            img_padded = np.zeros((img.shape[0], model_width, img.shape[2]))
            index_start_h = 0 
            index_start_w = int(abs(img.shape[1] - model_width) / 2.0)
            img_padded [:, index_start_w:index_start_w + img.shape[1], :] = img[:, :, :]  
        elif img.shape[0] < model_height and img.shape[1] < model_width:
            img_padded = np.zeros((model_height, model_width, img.shape[2]))
            index_start_h = int(abs(img.shape[0] - model_height) / 2.0)
            index_start_w = int(abs(img.shape[1] - model_width) / 2.0)
            img_padded [index_start_h:index_start_h + img.shape[0], index_start_w:index_start_w + img.shape[1], :] = img[:, :, :] 
        else:
            index_start_h = 0
            index_start_w = 0
            img_padded = np.copy(img)
        img = np.copy(img_padded)
        
        if use_patches:
            margin = int(self.margin_percent * model_width)
            width_mid = model_width - 2 * margin
            height_mid = model_height - 2 * margin
            img = img / float(255.0)
            img_h = img.shape[0]
            img_w = img.shape[1]
            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)
            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)
            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)
            
            for i in range(nxf):
                for j in range(nyf):
                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + model_width
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + model_width
                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + model_height
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + model_height
                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - model_width
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - model_height
                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                    label_p_pred = model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]), verbose=0)
                    seg = np.argmax(label_p_pred, axis=3)[0]
                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
                    
                    if i == 0 and j == 0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]
                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin, :] = seg_color
                    elif i == nxf-1 and j == nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]
                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0, :] = seg_color
                    elif i == 0 and j == nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]
                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin, :] = seg_color
                    elif i == nxf-1 and j == 0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]
                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0, :] = seg_color
                    elif i == 0 and j != 0 and j != nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]
                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin, :] = seg_color
                    elif i == nxf-1 and j != 0 and j != nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]
                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0, :] = seg_color
                    elif i != 0 and i != nxf-1 and j == 0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]
                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin, :] = seg_color
                    elif i != 0 and i != nxf-1 and j == nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]
                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin, :] = seg_color
                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]
                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin, :] = seg_color
            
            prediction_true = prediction_true[index_start_h: index_start_h+img_org_h, index_start_w: index_start_w+img_org_w,:]
            prediction_true = prediction_true.astype(np.uint8)
            
        else:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / float(255.0)
            img = resize_image(img, model_height, model_width)
            label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), verbose=0)
            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)
        return prediction_true[:, :, 0]
