# SPDX-License-Identifier: Apache-2.0
"""
PyTorch implementation of the SBB Binarization algorithm:
https://github.com/qurator-spk/eynollah/blob/main/src/eynollah/sbb_binarize.py

Conversion based on this fork:
https://github.com/twphl/sbb_binarizer_pytorch_converter/tree/main
"""
from pathlib import Path
import logging
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", module="onnx2torch")
logger: logging.Logger = logging.getLogger(__name__)


class TransposeWrapper(nn.Module):
    """
    Wrapper that handles input/output format conversion
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Input: NCHW [1, 3, 224, 448] -> Convert to NHWC for the wrapped model: [1, 224, 448, 3]
        x_nhwc = x.permute(0, 2, 3, 1)
        
        output = self.model(x_nhwc)

        # Output is in NHWC format [1, 224, 448, 2] -> Convert to NCHW: [1, 2, 224, 448]
        if isinstance(output, dict):
            output= list(output.values())[0]
        if len(output.shape) == 4:  # ty:ignore[unresolved-attribute]
            output = output.permute(0, 3, 1, 2)  # ty:ignore[unresolved-attribute]
        return output


class SbbBinarizer:
    """
    PyTorch implementation of SBB Binarization
    """
    def __init__(self, model: Path, device: str = 'auto') -> None:
        self.model_path: Path = model
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f'Using device: {self.device}')
        
        self.model, self.model_height, self.model_width, self.n_classes = self.load_model()
        logger.info(f'Model dimensions: {self.model_height}×{self.model_width}, {self.n_classes} classes')
        
    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path.as_posix(), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint: # Checkpoint format with metadata
                base_model = checkpoint['model']
                model_info = checkpoint.get('model_info', {})
            else: # Direct model format
                base_model = checkpoint
                model_info = {}
                
            wrapped_model = TransposeWrapper(base_model)
            wrapped_model.to(self.device)
            wrapped_model.eval()
            
            model_height = model_info.get('model_height', 224)
            model_width = model_info.get('model_width', 448)
            n_classes = model_info.get('output_channels', 2)

            return wrapped_model, model_height, model_width, n_classes
        except Exception as ex:
            logger.error(f'Failed to load model: {ex}')
            
    def __predict_patch(self, patch: np.ndarray) -> np.ndarray:
        patch = patch.astype(np.float32)
        if patch.max() > 1.5:   # uint8-like
            patch = patch / 255.0
        # Convert to tensor: HWC -> CHW -> NCHW
        # patch is (224, 448, 3) -> tensor (1, 3, 224, 448)
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(patch_tensor)
            # Output is now in NCHW format [1, 2, 224, 448] -> Get class predictions
            predictions = torch.argmax(output, dim=1)[0]  # Remove batch dimension -> [224, 448]
            predictions_numpy = predictions.cpu().numpy().astype(np.uint8)
            # Invert the predictions (0->1, 1->0) then convert to pixel values (0->255, 1->0)
            # predictions_numpy = (1 - predictions_numpy) * 255
        return predictions_numpy
    
    def __predict_patches(
        self, 
        img: np.ndarray, 
        index_start_h: int, 
        index_start_w: int, 
        img_org_h: int, 
        img_org_w: int, 
        n_batch_inference: int
    ) -> np.ndarray:
        margin = int(0.1 * self.model_width)
        width_mid = self.model_width - 2 * margin
        height_mid = self.model_height - 2 * margin

        img = img.astype(np.float32)
        img_h, img_w = img.shape[:2]

        prediction_true = np.zeros((img_h, img_w, 3))

        nxf = img_w / float(width_mid)
        nyf = img_h / float(height_mid)
        nxf = int(nxf) + 1 if nxf > int(nxf) else int(nxf)
        nyf = int(nyf) + 1 if nyf > int(nyf) else int(nyf)

        for i in range(nxf):
            for j in range(nyf):
                if i == 0:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + self.model_width
                else:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + self.model_width

                if j == 0:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + self.model_height
                else:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + self.model_height

                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - self.model_width
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - self.model_height

                patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                seg_in = self.__predict_patch(patch)
                seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)

                if i == 0 and j == 0:
                    seg_color = seg_color[
                        0 : seg_color.shape[0] - margin,
                        0 : seg_color.shape[1] - margin,
                        :
                    ]
                    prediction_true[
                        index_y_d + 0 : index_y_u - margin,
                        index_x_d + 0 : index_x_u - margin,
                        :
                    ] = seg_color
                elif i == nxf - 1 and j == nyf - 1:
                    seg_color = seg_color[
                        margin : seg_color.shape[0], 
                        margin : seg_color.shape[1], 
                        :
                    ]
                    prediction_true[
                        index_y_d + margin : index_y_u,
                        index_x_d + margin : index_x_u,
                        :
                    ] = seg_color
                elif i == 0 and j == nyf - 1:
                    seg_color = seg_color[
                        margin : seg_color.shape[0], 
                        0 : seg_color.shape[1] - margin, 
                        :
                    ]
                    prediction_true[
                        index_y_d + margin : index_y_u,
                        index_x_d + 0 : index_x_u - margin,
                        :
                    ] = seg_color
                elif i == nxf - 1 and j == 0:
                    seg_color = seg_color[
                        0 : seg_color.shape[0] - margin, 
                        margin : seg_color.shape[1], 
                        :
                    ]
                    prediction_true[
                        index_y_d + 0 : index_y_u - margin,
                        index_x_d + margin : index_x_u,
                        :
                    ] = seg_color
                elif i == 0 and j != 0 and j != nyf - 1:
                    seg_color = seg_color[
                        margin : seg_color.shape[0] - margin,
                        0 : seg_color.shape[1] - margin,
                        :
                    ]
                    prediction_true[
                        index_y_d + margin : index_y_u - margin,
                        index_x_d + 0 : index_x_u - margin,
                        :
                    ] = seg_color
                elif i == nxf - 1 and j != 0 and j != nyf - 1:
                    seg_color = seg_color[
                        margin : seg_color.shape[0] - margin,
                        margin : seg_color.shape[1],
                        :
                    ]
                    prediction_true[
                        index_y_d + margin : index_y_u - margin,
                        index_x_d + margin : index_x_u,
                        :
                    ] = seg_color
                elif i != 0 and i != nxf - 1 and j == 0:
                    seg_color = seg_color[
                        0 : seg_color.shape[0] - margin,
                        margin : seg_color.shape[1] - margin,
                        :
                    ]
                    prediction_true[
                        index_y_d + 0 : index_y_u - margin,
                        index_x_d + margin : index_x_u - margin,
                        :
                    ] = seg_color
                elif i != 0 and i != nxf - 1 and j == nyf - 1:
                    seg_color = seg_color[
                        margin : seg_color.shape[0],
                        margin : seg_color.shape[1] - margin,
                        :
                    ]
                    prediction_true[
                        index_y_d + margin : index_y_u,
                        index_x_d + margin : index_x_u - margin,
                        :
                    ] = seg_color
                else:
                    seg_color = seg_color[
                        margin : seg_color.shape[0] - margin,
                        margin : seg_color.shape[1] - margin,
                        :
                    ]
                    prediction_true[
                        index_y_d + margin : index_y_u - margin,
                        index_x_d + margin : index_x_u - margin,
                        :
                    ] = seg_color

        prediction_true = prediction_true[
            index_start_h : index_start_h + img_org_h,
            index_start_w : index_start_w + img_org_w,
            :
        ]
        return prediction_true[:, :, 0].astype(np.uint8)
    
    def __predict_image(self, img: np.ndarray, img_org_h: int, img_org_w: int):
        img = img.astype(np.float32) / 255.0
        img_resized = cv2.resize(
            img, 
            (self.model_width, self.model_height), 
            interpolation=cv2.INTER_NEAREST
        )
        prediction = self.__predict_patch(img_resized)
        seg_color = np.repeat(prediction[:, :, np.newaxis], 3, axis=2)
        prediction_true = cv2.resize(
            seg_color, 
            (img_org_w, img_org_h), 
            interpolation=cv2.INTER_NEAREST
        )
        return prediction_true[:, :, 0].astype(np.uint8)
        
    def __predict(self, img: np.ndarray, use_patches: bool = True, n_batch_inference: int = 5) -> np.ndarray:
        img_org_h, img_org_w = img.shape[:2]
        
        if img.shape[0] < self.model_height and img.shape[1] >= self.model_width:
            img_padded = np.zeros((self.model_height, img.shape[1], img.shape[2]))
            index_start_h = int(abs(img.shape[0] - self.model_height) / 2.0)
            index_start_w = 0
            img_padded[index_start_h : index_start_h + img.shape[0], :, :] = img[:, :, :]

        elif img.shape[0] >= self.model_height and img.shape[1] < self.model_width:
            img_padded = np.zeros((img.shape[0], self.model_width, img.shape[2]))
            index_start_h = 0
            index_start_w = int(abs(img.shape[1] - self.model_width) / 2.0)
            img_padded[:, index_start_w : index_start_w + img.shape[1], :] = img[:, :, :]

        elif img.shape[0] < self.model_height and img.shape[1] < self.model_width:
            img_padded = np.zeros((self.model_height, self.model_width, img.shape[2]))
            index_start_h = int(abs(img.shape[0] - self.model_height) / 2.0)
            index_start_w = int(abs(img.shape[1] - self.model_width) / 2.0)
            img_padded[index_start_h : index_start_h + img.shape[0], index_start_w : index_start_w + img.shape[1], :] = img[:, :, :]

        else:
            index_start_h = 0
            index_start_w = 0
            img_padded = np.copy(img)

        img = np.copy(img_padded)
        
        if use_patches:
            return self.__predict_patches(img, index_start_h, index_start_w, img_org_h, img_org_w, n_batch_inference)
        else:
            return self.__predict_image(img, img_org_h, img_org_w)
    
    def run(self, image: np.ndarray, use_patches: bool = True) -> np.ndarray:
        res = self.__predict(image, use_patches).astype(np.uint8)
        
        out = (1 - res) * 255   # -> 0/255
        return out.astype(np.uint8)
