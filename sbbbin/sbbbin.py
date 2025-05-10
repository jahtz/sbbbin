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
# This file includes code from the eynollah project,
# available at https://github.com/qurator-spk/eynollah and licensed under
# Apache 2.0 license https://github.com/qurator-spk/eynollah/blob/main/LICENSE

import sys
from pathlib import Path
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as tensorflow_backend
sys.stderr = stderr
cv2.setLogLevel(2)
tf.get_logger().setLevel(logging.ERROR)


def resize_image(image, height, width):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


class SbbBinarizer:
    def __init__(self, model: Path, gpu: bool = True):
        self.start_new_session(gpu)
        if not model.exists():
            raise ValueError(f"Model does not exist in {model}")
        self.model = self.load_model(model)

    def start_new_session(self, gpu: bool = True):
        if not gpu:
            tf.config.set_visible_devices([], "GPU")
        elif not tf.config.list_physical_devices("GPU"):
            raise EnvironmentError("Could not find any valid GPUs. Fallback to CPU computation.")
            raise 
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
        tensorflow_backend.set_session(self.session)

    def end_session(self):
        tensorflow_backend.clear_session()
        self.session.close()
        del self.session

    def load_model(self, model):
        model = load_model(model.as_posix(), compile=False)
        model_height = model.layers[len(model.layers)-1].output_shape[1]
        model_width = model.layers[len(model.layers)-1].output_shape[2]
        n_classes = model.layers[len(model.layers)-1].output_shape[3]
        return model, model_height, model_width, n_classes

    def predict(self, model: tuple, img, n_batch_inference=5):
        tensorflow_backend.set_session(self.session)
        model, model_height, model_width, n_classes = model
        img_org_h = img.shape[0]
        img_org_w = img.shape[1]
        
        if img.shape[0] < model_height and img.shape[1] >= model_width:
            img_padded = np.zeros((model_height, img.shape[1], img.shape[2]))
            index_start_h = int(abs(img.shape[0] - model_height) / 2.)
            index_start_w = 0
            img_padded [index_start_h:index_start_h+img.shape[0], :, :] = img[:, :, :]
        elif img.shape[0] >= model_height and img.shape[1] < model_width:
            img_padded = np.zeros((img.shape[0], model_width, img.shape[2]))
            index_start_h =  0 
            index_start_w = int(abs(img.shape[1] - model_width) / 2.)
            img_padded [:, index_start_w:index_start_w+img.shape[1], :] = img[:, :, :]
        elif img.shape[0] < model_height and img.shape[1] < model_width:
            img_padded = np.zeros((model_height, model_width, img.shape[2]))
            index_start_h = int(abs(img.shape[0] - model_height) / 2.)
            index_start_w = int(abs(img.shape[1] - model_width) / 2.)
            img_padded [index_start_h:index_start_h+img.shape[0], index_start_w:index_start_w+img.shape[1], :] = img[:, :, :]
        else:
            index_start_h = 0
            index_start_w = 0
            img_padded = np.copy(img)
             
        img = np.copy(img_padded)
        margin = int(0.1 * model_width)
        width_mid = model_width - 2 * margin
        height_mid = model_height - 2 * margin
        img = img / float(255.0)
        img_h = img.shape[0]
        img_w = img.shape[1]
        prediction_true = np.zeros((img_h, img_w, 3))
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
            
        list_i_s = []
        list_j_s = []
        list_x_u = []
        list_x_d = []
        list_y_u = []
        list_y_d = []
        
        img_patch = np.zeros((n_batch_inference, model_height, model_width, 3))
        batch_indexer = 0
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
                list_i_s.append(i)
                list_j_s.append(j)
                list_x_u.append(index_x_u)
                list_x_d.append(index_x_d)
                list_y_d.append(index_y_d)
                list_y_u.append(index_y_u)
                img_patch[batch_indexer, :, :, :] = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                batch_indexer += 1
                
                if batch_indexer == n_batch_inference:
                    label_p_pred = model.predict(img_patch, verbose=0)
                    seg = np.argmax(label_p_pred, axis=3)                    
                    indexer_inside_batch = 0
                    for i_batch, j_batch in zip(list_i_s, list_j_s):
                        seg_in = seg[indexer_inside_batch, :, :]
                        seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                        index_y_u_in = list_y_u[indexer_inside_batch]
                        index_y_d_in = list_y_d[indexer_inside_batch]
                        index_x_u_in = list_x_u[indexer_inside_batch]
                        index_x_d_in = list_x_d[indexer_inside_batch]
                        if i_batch == 0 and j_batch == 0:
                            seg_color = seg_color[0:seg_color.shape[0]-margin, 0:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+0:index_y_u_in-margin, index_x_d_in+0:index_x_u_in - margin, :] = seg_color
                        elif i_batch == nxf - 1 and j_batch == nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-0, margin:seg_color.shape[1]-0, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-0, index_x_d_in+margin:index_x_u_in-0, :] = seg_color
                        elif i_batch == 0 and j_batch == nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-0, 0:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-0, index_x_d_in+0:index_x_u_in-margin, :] = seg_color
                        elif i_batch == nxf - 1 and j_batch == 0:
                            seg_color = seg_color[0:seg_color.shape[0]-margin, margin:seg_color.shape[1]-0, :]
                            prediction_true[index_y_d_in+0:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-0, :] = seg_color
                        elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-margin, 0:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-margin, index_x_d_in+0:index_x_u_in-margin, :] = seg_color
                        elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-margin, margin:seg_color.shape[1]-0, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-0, :] = seg_color
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                            seg_color = seg_color[0:seg_color.shape[0]-margin, margin:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+0:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-margin, :] = seg_color
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-0, margin:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-0, index_x_d_in+margin:index_x_u_in-margin, :] = seg_color
                        else:
                            seg_color = seg_color[margin:seg_color.shape[0]-margin, margin:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-margin, :] = seg_color
                        indexer_inside_batch += 1        
                    list_i_s = []
                    list_j_s = []
                    list_x_u = []
                    list_x_d = []
                    list_y_u = []
                    list_y_d = []
                    batch_indexer = 0
                    img_patch = np.zeros((n_batch_inference, model_height, model_width, 3))
                    
                elif i == (nxf - 1) and j == (nyf - 1):
                    label_p_pred = model.predict(img_patch, verbose=0)
                    seg = np.argmax(label_p_pred, axis=3)               
                    indexer_inside_batch = 0
                    for i_batch, j_batch in zip(list_i_s, list_j_s):
                        seg_in = seg[indexer_inside_batch, :, :]
                        seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                        index_y_u_in = list_y_u[indexer_inside_batch]
                        index_y_d_in = list_y_d[indexer_inside_batch]
                        index_x_u_in = list_x_u[indexer_inside_batch]
                        index_x_d_in = list_x_d[indexer_inside_batch]
                        
                        if i_batch == 0 and j_batch == 0:
                            seg_color = seg_color[0:seg_color.shape[0]-margin, 0:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+0:index_y_u_in-margin, index_x_d_in+0:index_x_u_in-margin, :] = seg_color
                        elif i_batch == nxf - 1 and j_batch == nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-0, margin:seg_color.shape[1]-0, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-0, index_x_d_in+margin:index_x_u_in-0, :] = seg_color
                        elif i_batch == 0 and j_batch == nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-0, 0:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-0, index_x_d_in+0:index_x_u_in-margin, :] = seg_color
                        elif i_batch == nxf - 1 and j_batch == 0:
                            seg_color = seg_color[0:seg_color.shape[0]-margin, margin:seg_color.shape[1]-0, :]
                            prediction_true[index_y_d_in+0:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-0, :] = seg_color
                        elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-margin, 0:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-margin, index_x_d_in+0:index_x_u_in-margin, :] = seg_color
                        elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-margin, margin:seg_color.shape[1]-0, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-0, :] = seg_color
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                            seg_color = seg_color[0:seg_color.shape[0]-margin, margin:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+0:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-margin, :] = seg_color
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                            seg_color = seg_color[margin:seg_color.shape[0]-0, margin:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-0, index_x_d_in+margin:index_x_u_in-margin, :] = seg_color
                        else:
                            seg_color = seg_color[margin:seg_color.shape[0]-margin, margin:seg_color.shape[1]-margin, :]
                            prediction_true[index_y_d_in+margin:index_y_u_in-margin, index_x_d_in+margin:index_x_u_in-margin, :] = seg_color
                        indexer_inside_batch += 1
                                
                    list_i_s = []
                    list_j_s = []
                    list_x_u = []
                    list_x_d = []
                    list_y_u = []
                    list_y_d = []
                    batch_indexer = 0
                    img_patch = np.zeros((n_batch_inference, model_height, model_width,3))
        
        prediction_true = prediction_true[index_start_h:index_start_h+img_org_h, index_start_w:index_start_w+img_org_w, :]
        prediction_true = prediction_true.astype(np.uint8)
        return prediction_true[:, :, 0]

    def run(self, image: Path):
        image = cv2.imread(image.as_posix())
        res = self.predict(self.model, image)
        res[res == 0] = 2
        res = (res - 1) * 255
        return res.astype(np.uint8)
