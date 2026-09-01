# 03 · YOLO 数据与训练

YOLO 负责**感知**：在图像里框出「方块」和「目标框」。本项目训练了 3 个类别：

| 类别名 | 含义 |
|---|---|
| `red_fang` | 红色方块（可抓） |
| `brown_fang` | 棕色方块（可抓） |
| `bule_box` | 蓝色目标框（要放进去） |

> 以后想加类别，只要：① 重新标注训练 ② 把新类名加进 `step7_pick_place.py` 的 `CUBE_CLASSES` / `BOX_CLASSES`。

## 3.1 Ubuntu 上采集数据（step5_collect_yolo.py）

```bash
python step5_collect_yolo.py
```

- 空格保存、q 退出，照片存到 `datasets/raw/`
- **300~600 张**，变化越多泛化越好：
  - 方块换**多种颜色、多种大小**
  - 方块和框放**不同位置、不同距离**
  - **不同光照**（开灯/关灯/拉窗帘）、桌上放杂物
  - 机械臂 / 相机换不同高度角度

## 3.2 Windows 上标注（labelImg）

把 `datasets/raw/` 拷到 Windows：

```powershell
pip install ultralytics labelImg
labelImg
```

1. `Open Dir` 选图片文件夹；`Change Save Dir` 选 `labels` 文件夹
2. 右上角把 `PascalVOC` 切换成 **`YOLO`**
3. 画框：方块标 `red_fang` / `brown_fang`，框标 `bule_box`
4. `Ctrl+S` 保存，`D` 下一张

> 用 **seg（分割）** 还是 detect？本项目用 `yolo segment`（带掩码），方便后面算方块朝向。如果嫌标注麻烦，detect 版也能跑（朝向固定为 0）。

## 3.3 整理数据集 + 训练（Windows）

目录结构：

```
cube_box/
├── images/
│   ├── train/   (约 80%)
│   └── val/     (约 20%)
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

`dataset.yaml`：

```yaml
path: C:/你的路径/cube_box
train: images/train
val: images/val
names:
  0: red_fang
  1: bule_box
  2: brown_fang
```

训练（有 NVIDIA 显卡快很多）：

```powershell
cd cube_box
yolo segment train data=dataset.yaml model=yolo26n-seg.pt epochs=200 imgsz=640 batch=16 device=0
```

- 显存不足：`batch=8` 或更小
- 没显卡：去掉 `device=0`（会很慢，先用 `epochs=50 imgsz=480` 试跑通）

## 3.4 部署到 Ubuntu

```bash
mkdir -p ~/robot_learning/pick_place/weights
# 把 best.pt 拷到 ~/robot_learning/pick_place/weights/best.pt
```

## 3.5 预览检测效果（step6_detect_preview.py）

```bash
python step6_detect_preview.py
```

看到 `red_fang` / `brown_fang` / `bule_box` 的绿色框就成功了。

## 3.6 常见问题

| 问题 | 解决 |
|---|---|
| 检测不到 / 框乱跳 | 数据太少或变化不够；把 `step6` / `step7` 的 `conf` 从 0.3 调低到 0.2 |
| 把杂物也框成方块 | 数据里多放杂物并标对 |
| `CUDA out of memory` | `batch` 调小 |
| 中文类名 / 特殊字符 | YOLO 类名建议用英文小写+下划线 |

---

下一步 → [04 · 主程序与调参](04-主程序与调参.md)

