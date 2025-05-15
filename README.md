# 🫁 医学图像三维可视化工具（PyQt + PyVista）

本项目为医学影像（如 `.nii.gz`）提供三维重建、标签叠加、体积统计和截图导出功能。  
基于 PyQt5 + PyVista 构建，适用于科研演示与医学图像分析学习。

---

## 🎯 功能特性

- ✅ 图形界面（PyQt5）
- ✅ 三维体绘渲染（PyVista）
- ✅ 自动识别图像与标签
- ✅ 标签可开关
- ✅ 自动统计每个标签体积（单位：mm³）
- ✅ 支持截图导出

---

## 🧪 数据来源说明

本项目使用的数据来自：

**Medical Segmentation Decathlon（MSD） - Task06_Lung**

- 🔗 官网主页：[https://medicaldecathlon.com](https://medicaldecathlon.com)
- 🔗 AWS 镜像链接（可下载）：[https://medicaldecathlon.com/dataaws/](https://medicaldecathlon.com/dataaws/)
- 数据内容：肺部 CT 扫描图像及其肿瘤区域标签（`.nii.gz` 格式）
- 标签含义：
  - `0`：背景
  - `1`：肿瘤区域

> 数据版权归 MSD 项目方所有，仅供科研与教学用途。

---

## 🚀 快速上手

1. 安装依赖

```bash
pip install -r requirements.txt
