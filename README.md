# 跨模态文本-图像检索项目 README
## 1. 项目简介
这是一个基于 CLIP (Contrastive Language-Image Pre-training) 模型的简单跨模态检索系统。项目的核心功能是让用户输入一段文本，系统能从预先准备好的图片数据集中找到并展示出与该文本描述最匹配的图片。

该项目代码结构清晰，易于理解和扩展，采用了模块化的设计，将配置、数据处理、模型管理和主程序逻辑分别存放在不同的 Python 文件中。

## 2. 功能特点
文本到图像检索： 根据自然语言描述检索相关图片。

使用预训练模型： 利用强大的预训练 CLIP 模型（其实自己没有训练过，这个效果就会很差，所以我准备之后再换成自己专门在coco数据上训练好的模型），无需自己进行训练。

模块化设计： 代码分为多个 .py 文件，方便维护和升级。


COCO 数据集支持： 使用真实的 COCO 2017 数据集进行演示，更贴近实际应用场景。

## 3. 项目结构
```
.
├── main.py                 # 项目主程序，负责流程调度和用户交互
├── model_handler.py        # 封装了 CLIP 模型的加载和编码逻辑
├── data_preparer.py        # 负责 COCO 数据集的加载和处理
├── retriever_config.py     # 存放所有项目配置，如模型名、数据路径等
└── README.md
```
## 4. 环境配置
本项目需要 Python 3.6+ 环境。请使用以下命令安装所有依赖库：
```
pip install torch torchvision torchaudio transformers pillow matplotlib
```

## 5. 数据集准备
本项目使用 COCO 2017 数据集进行演示。你需要下载以下两个文件（也可以不用这个，不用的话改一下代码就好了）：

图片数据集： train2017.zip 或 val2017.zip

标注文件： annotations_trainval2017.zip

然后，在 retriever_config.py 文件中，将 COCO_DATA_ROOT 路径修改为你的 COCO 数据集根目录路径。

## 6. 如何运行
在配置好环境和数据后，只需在终端中运行主程序：
```
python main.py
```
程序启动后，会提示你输入文本查询。
\
首次运行： 程序将自动从网络下载 CLIP 模型权重，这可能需要一些时间（约几百 MB）。下载一次后，模型会缓存到本地，下次运行会快很多。
\
程序运行中： 输入任何你想要检索的文本描述，比如 "a man holding a surfboard" 或 "A cat sleeping on a sofa"。程序会返回并显示最相关的图片。
\
退出程序： 输入 exit 并按回车。

## 7. 自定义与扩展
修改图片数量： 在 retriever_config.py 中，修改 MAX_IMAGES_TO_PROCESS 的值来改变处理的图片数量。将其设置为 None 则处理所有图片。

更换模型： 在 retriever_config.py 中，你可以将 MODEL_NAME 更改为其他 CLIP 模型，例如 openai/clip-vit-large-patch14 以获得更高的性能。

使用自定义数据集： 如果你想使用自己的图片集，可以修改 data_preparer.py，使其读取你的图片文件和对应的描述信息。

部署为 Web 服务： 将 main.py 的核心逻辑封装成 API，使用 Flask 或 Streamlit 框架可以轻松将其部署为一个可视化的 Web 应用。
