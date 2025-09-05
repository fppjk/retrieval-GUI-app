#这个文件存储了所有的常量和配置，方便集中管理。

# CLIP 模型名称
# "openai/clip-vit-base-patch32" 或 "openai/clip-vit-large-patch14"
MODEL_NAME = "openai/clip-vit-base-patch32"

# 存放示例图片的文件夹路径
IMAGE_FOLDER = "sample_images"

# 示例图片的描述，用于生成图片
IMAGE_DESCRIPTIONS = {
    "red_apple.png": "A vibrant red apple",
    "blue_car.png": "A sleek blue car",
    "green_forest.png": "A lush green forest",
    "yellow_sun.png": "A bright yellow sun",
    "black_cat.png": "A mysterious black cat",
}

# COCO 数据集路径配置
COCO_DATA_ROOT = "F:/Lab/Coco"
COCO_IMAGES_DIR = "val2017" 
COCO_CAPTIONS_FILE = "annotations/captions_val2017.json"

# 限制处理的图片数量，设置为 None 则处理所有图片
MAX_IMAGES_TO_PROCESS = 100

# 统一的模型处理器
# 在此处指定要使用的模型处理器类
MODEL_HANDLER_CLASS = 'CLIPModelHandler'