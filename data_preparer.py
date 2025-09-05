# 读取 COCO 的 JSON 文件，并整理出图片路径和文字描述的对应关系。
# data_preparer.py
import os
import json

def prepare_coco_data(data_root, images_dir, captions_file, max_images=None):
    """
    加载 COCO 数据集，返回图片路径和对应的文字描述。
    
    Args:
        data_root (str): COCO 数据集根目录路径。
        images_dir (str): 图片子目录名称 (如 'train2017')。
        captions_file (str): 标注文件路径 (如 'annotations/captions_train2017.json')。
        max_images (int, optional): 限制处理的图片数量。默认为 None，表示不限制。
    """
    print("--- 正在加载 COCO 数据集 ---")
    captions_path = os.path.join(data_root, captions_file)
    images_dir_path = os.path.join(data_root, images_dir)

    if not os.path.exists(captions_path) or not os.path.exists(images_dir_path):
        raise FileNotFoundError(f"未找到 COCO 数据集文件。请确保 {captions_path} 和 {images_dir_path} 存在。")

    with open(captions_path, 'r') as f:
        coco_data = json.load(f)

    # 构建从 image_id 到其文件名的映射
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # 提取每张图片的第一条文字描述
    image_paths = []
    image_descriptions = []
    processed_images = set()

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in processed_images:
            filename = image_id_to_filename[image_id]
            image_path = os.path.join(images_dir_path, filename)
            image_paths.append(image_path)
            image_descriptions.append(ann['caption'])
            processed_images.add(image_id)
            
            # 检查是否达到数量限制
            if max_images and len(image_paths) >= max_images:
                break
            
    print(f"已加载 {len(image_paths)} 张图片及其描述。")
    return image_paths, image_descriptions