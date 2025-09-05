# 本文件为程序入口，负责调用项目内的所有模块，组织程序的流程。
import torch
import matplotlib.pyplot as plt
from retriever_config import MODEL_NAME, COCO_DATA_ROOT, COCO_IMAGES_DIR, COCO_CAPTIONS_FILE, MAX_IMAGES_TO_PROCESS
from data_preparer import prepare_coco_data
from model_handler import CLIPHandler
import os
from PIL import Image

def main():
    """
    主函数：执行COCO 数据集的 跨模态检索(Cross-modal Retrieval) 程序。
    """
    # --- 添加以下代码来设置 Matplotlib 的字体 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei (微软雅黑也可以：'Microsoft YaHei')
    plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题
    # -----------------------------------------------

    # 1. 准备数据
    image_paths, image_descriptions = prepare_coco_data(
        COCO_DATA_ROOT, COCO_IMAGES_DIR, COCO_CAPTIONS_FILE, max_images=MAX_IMAGES_TO_PROCESS
    )

    # 2. 加载模型
    try:
        clip_handler = CLIPHandler(MODEL_NAME)
        print("模型加载成功。")
    except RuntimeError as e:
        print(f"错误: {e}")
        return
    
    # 3. 编码图片
    image_features, images = clip_handler.encode_images(image_paths)
    
    # 归一化图片特征向量
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)




    # 4. 启动交互式检索循环
    print("\n--- 启动跨模态检索程序 ---")
    print("请输入文本查询，或输入 'exit' 退出。")

    while True:
        query_text = input("\n您的查询: ")
        if query_text.lower() == 'exit':
            break

        # 编码文本
        text_features = clip_handler.encode_text(query_text)
        
        # 归一化文本特征向量
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarities = (text_features_norm @ image_features_norm.T).squeeze(0)

        # 找到最相似的图片
        best_match_idx = torch.argmax(similarities).item()
        best_match_score = similarities[best_match_idx].item()

        # 获取最匹配图片的描述和文件名
        best_match_path = image_paths[best_match_idx]
        best_match_caption = image_descriptions[best_match_idx]


        print(f"最匹配的图片是: {os.path.basename(best_match_path)}")
        print(f"原始描述: {best_match_caption}")
        print(f"相似度: {best_match_score:.4f}")

        # 展示最匹配的图片
        plt.imshow(images[best_match_idx])
        plt.title(f"最匹配图片: {os.path.basename(best_match_path)}\n原始描述: {best_match_caption}\n相似度: {best_match_score:.4f}")
        plt.axis('off')
        plt.show()

    print("程序已退出。")

if __name__ == "__main__":
    main()