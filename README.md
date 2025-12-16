# ai-clip-editor — 錒i剪辑师：时刻检索 Baseline

简介
- 最小可用实现（baseline）：使用 CLIP 图像 encoder 对视频按时间窗口提取向量，建立 Faiss 向量索引。给定自然语言查询后检索并合并相邻高分窗口得到时间段（start, end）。

仓库结构
- README.md
- requirements.txt
- .gitignore
- scripts/
  - utils.py
  - extract_features.py
  - build_index.py
  - retrieve_moments.py

在 Google Colab 运行（推荐，适合无显卡的情况）
1. 打开 https://colab.research.google.com → 新建 Notebook → Runtime → Change runtime type → Hardware accelerator 选择 GPU（可选）。
2. 依次运行下面的 cell（先安装依赖，再把脚本写入 Colab 文件系统并运行）：
   - 安装依赖：
     ```
     !pip install -q torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
     !pip install -q transformers ftfy tqdm opencv-python-headless numpy faiss-cpu scipy
     ```
   - 将脚本写入 /content/ai-clip-editor/scripts（我已把脚本写法整理在仓库，你也可以 clone 本仓库到 Colab）
   - 下载 demo.mp4（或在 Colab 上传你自己的 demo.mp4）
   - 运行提取特征、建索引、检索脚本（示例见仓库 README）

快速开始（本地 / Colab）
- 抽特征（示例）：
  python scripts/extract_features.py --video demo.mp4 --out demo_features.npz --window_sec 2 --stride_sec 1 --fps 1 --device cuda
- 建索引：
  python scripts/build_index.py --features demo_features.npz --index demo.index
- 检索：
  python scripts/retrieve_moments.py --index demo.index --features demo_features_norm.npz --query "把杯子放到桌子上" --device cuda

注意
- Colab 初次下载 transformers 模型会花时间与流量。
- CLIP 的文本 encoder 对中文支持有限，临时可先把中文查询翻译为英文，或后续替换为中文 multimodal 模型。

我可以继续帮你：
- 把这些文件为你直接 push 到远程（需要你在本地运行一条 gh 登录命令或给我授权），或
- 直接把一个可运行的 Colab notebook (.ipynb) 给你（你可以直接打开运行），或
- 现在陪你在 GitHub 网页里逐个创建文件（你按提示我在每步等待确认）。
