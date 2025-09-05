# 这个文件封装了所有与 CLIP 模型交互的逻辑，包括加载模型和编码数据。
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from abc import ABC, abstractmethod
import os

# 定义通用的模型处理器抽象基类
class BaseModelHandler(ABC):
    """
    所有模型处理器的抽象基类。
    """
    def __init__(self, model_name):
        self.model_name = model_name
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 GPU。此程序需要 GPU 以进行高效计算。")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def encode_images(self, image_paths, batch_size=32):
        """
        编码图片列表，返回图片的特征向量和 PIL.Image 对象。
        """
        pass

    @abstractmethod
    def encode_text(self, text):
        """
        编码文本，返回文本的特征向量。
        """
        pass

# 实现 CLIP 的具体处理器类
class CLIPModelHandler(BaseModelHandler):
    """
    负责 CLIP 模型的加载和数据编码。
    """
    def __init__(self, model_name):
        super().__init__(model_name)
        
        print("--- 正在加载 CLIP 模型和处理器 ---")
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 GPU。此程序需要 GPU 以进行高效计算。")
        self.device = "cuda"
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"CLIP 模型已加载到 {self.device}.")

    def encode_images(self, image_paths, batch_size=32):
        print("\n--- 正在编码所有图片 ---")
        all_features = []
        all_images = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(pixel_values=inputs.pixel_values)
                all_features.append(features)
                all_images.extend(batch_images)
            
            print(f"已处理 {min(i + batch_size, len(image_paths))} / {len(image_paths)} 张图片...", end='\r')

        print(f"\n已编码 {len(all_images)} 张图片。")
        return torch.cat(all_features, dim=0), all_images

    def encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(input_ids=inputs.input_ids)
        return features

# 如果未来要添加其他模型，只需要创建一个新的类，继承 BaseModelHandler 并实现其方法即可
# 例如：
# class AnotherModelHandler(BaseModelHandler):
#     def __init__(self, model_name):
#         super().__init__(model_name)
#         # 加载你的模型和处理器
#     def encode_images(self, image_paths, batch_size=32):
#         # 实现图片编码逻辑
#     def encode_text(self, text):
#         # 实现文本编码逻辑