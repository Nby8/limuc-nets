'''
Author: nby 2403050046@qq.com
Date: 2025-04-11 14:09:49
LastEditors: nby 2403050046@qq.com
LastEditTime: 2025-04-11 15:07:24
FilePath: \ViT-GradCAM-main\gradcam.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations

class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()  # Set the model to evaluation mode
        self.feature = None  # To store the features from the target layer
        self.gradient = None  # To store the gradients from the target layer
        self.handlers = []  # List to keep track of hooks
        self.target = target  # 指定的目标层
        self._get_hook()  # 为目标层注册钩子

    # Hook to get features from the forward pass 前向传播钩子
    def _get_features_hook(self, module, input, output):      
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    # Hook to get gradients from the backward pass 反向传播钩子
    def _get_grads_hook(self, module, input_grad, output_grad):        
        self.gradient = self.reshape_transform(output_grad)  # Store and reshape the output gradients

        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)  # Store gradients for later use

        output_grad.register_hook(_store_grad)  # Register hook to store gradients

    # 注册钩子
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    # Function to reshape the tensor for visualization
    def reshape_transform(self, tensor, height=14, width=14):
        '''
        tensor: 输入张量， Transformer模型中某一层的输出特征图（batch_size, seq_len, embed_dim)
        '''
        
        # 如果输入是四维张量 (batch_size, channels, height, width)，直接使用
        if len(tensor.shape) == 4:
            return tensor  # 无需 reshape，直接返回
        
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))       # 将序列长度重排列为二维网格（H×W), 保留嵌入维度
        result = result.transpose(2, 3).transpose(1, 2)  # （batch_size, embed_dim, height, width) --> (B,C,H,W)
        return result

    # Function to compute the Grad-CAM heatmap
    def __call__(self, inputs):
        self.model.zero_grad()  # Zero the gradients       清零梯度
        output = self.model(inputs)  # Forward pass        前向传播

        # Get the index of the highest score in the output
        index = np.argmax(output.cpu().data.numpy())           # 获取模型输出中得分最高的类别索引
        target = output[0][index]  # Get the target score           提取该类别的得分，用于计算梯度
        target.backward()  # Backward pass to compute gradients     反向传播，计算梯度

        # Get the gradients and features
        gradient = self.gradient[0].cpu().data.numpy()           # 通过钩子捕获的目标层梯度
        weight = np.mean(gradient, axis=(1, 2))  # Average the gradients
        feature = self.feature[0].cpu().data.numpy()            # 通过钩子捕获的目标层特征图

        # Compute the weighted sum of the features
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)  # Sum over the channels       对所有通道求和
        cam = np.maximum(cam, 0)  # Apply ReLU to remove negative values

        # Normalize the heatmap
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))  # Resize to match the input image size
        return cam  # Return the Grad-CAM heatmap
