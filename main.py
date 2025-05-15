import sys
import os
import numpy as np
import nibabel as nib
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QFileDialog, QPushButton, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QDateTime


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学图像三维可视化 + 体积统计")
        self.resize(1280, 960)
        self.setStyleSheet("background-color: #f0f0f0;")

        # 中央窗口容器
        self.frame = QWidget()
        self.setCentralWidget(self.frame)

        # 嵌入 PyVista 视图
        self.plotter = QtInteractor(self.frame)
        self.plotter.setStyleSheet("border: 1px solid #ccc;")

        # 控制按钮区域
        self.open_button = QPushButton("📂 打开图像")
        self.open_button.setStyleSheet("padding: 8px; font-weight: bold;")
        self.open_button.clicked.connect(self.select_image_and_label)

        self.toggle_label_button = QPushButton("👁 隐藏标签")
        self.toggle_label_button.setEnabled(False)
        self.toggle_label_button.setStyleSheet("padding: 8px; font-weight: bold;")
        self.toggle_label_button.clicked.connect(self.toggle_label_visibility)

        self.save_png_button = QPushButton("📸 导出截图")
        self.save_png_button.setEnabled(False)
        self.save_png_button.setStyleSheet("padding: 8px; font-weight: bold;")
        self.save_png_button.clicked.connect(self.save_screenshot)

        # 标签可见性状态
        self.label_visible = True
        self.label_actor = None
        self.grid_label = None

        # 体积信息展示
        self.volume_label = QLabel("<b>体积统计信息：</b>")
        self.volume_label.setWordWrap(True)
        self.volume_label.setTextFormat(Qt.RichText)
        self.volume_label.setFont(QFont("Arial", 10))
        self.volume_label.setStyleSheet("padding: 8px; background-color: #ffffff; border: 1px solid #ccc;")

        # 按钮区域布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.toggle_label_button)
        button_layout.addWidget(self.save_png_button)

        # 主布局
        layout = QVBoxLayout()
        layout.addWidget(self.plotter.interactor, stretch=4)
        layout.addLayout(button_layout)
        layout.addWidget(self.volume_label, stretch=1)
        self.frame.setLayout(layout)

    def select_image_and_label(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "NIfTI Files (*.nii *.nii.gz)")
        if not img_path:
            return

        # 自动推测标签路径
        base_dir = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        label_dir = base_dir.replace("imagesTr", "labelsTr")
        label_path = os.path.join(label_dir, filename)

        if not os.path.exists(label_path):
            self.statusBar().showMessage(f"未找到标签文件：{label_path}", 5000)
            return

        self.load_and_render(img_path, label_path)

    def load_and_render(self, img_path, label_path):
        self.plotter.clear()

        img = nib.load(img_path)
        label = nib.load(label_path)
        img_data = img.get_fdata().astype(np.float32)
        label_data = label.get_fdata().astype(np.uint8)

        # 归一化图像
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        spacing = img.header.get_zooms()[:3]
        dims = np.array(img_data.shape) + 1

        # 创建图像体数据
        grid_img = pv.ImageData(dimensions=dims)
        grid_img.spacing = spacing
        grid_img.origin = (0, 0, 0)
        grid_img.cell_data["values"] = img_data.flatten(order="F")

        # 创建标签体数据
        self.grid_label = pv.ImageData(dimensions=dims)
        self.grid_label.spacing = spacing
        self.grid_label.origin = (0, 0, 0)
        self.grid_label.cell_data["label"] = label_data.flatten(order="F")

        # 渲染体绘制
        # self.plotter.add_volume(grid_img, scalars="values", cmap="bone", opacity="sigmoid", shade=True)

        self.plotter.add_volume(
            grid_img,
            scalars="values",
            cmap="bone",
            opacity="sigmoid",
            shade=True,
            show_scalar_bar=False  # ❗不要自动加 scalar_bar
        )

        self.plotter.add_scalar_bar(
            title="values",
            vertical=False,  # ✅ 横向
            position_x=0.3,  # 起点靠左
            position_y=0.05,  # 距离底部高度
            width=0.4,  # 宽度占整个窗口 40%
            height=0.05,  # 高度占 5%
            title_font_size=15,
            label_font_size=13,
            n_labels=4
        )

        self.label_actor = self.plotter.add_volume(
            self.grid_label, scalars="label", cmap="Reds",
            opacity=[0, 0, 0.1, 0.4, 0.6, 0.8, 1.0], show_scalar_bar=False
        )

        self.plotter.reset_camera()

        self.toggle_label_button.setEnabled(True)
        self.save_png_button.setEnabled(True)
        self.label_visible = True

        # 输出体积统计
        self.show_volume_stats(label_data, spacing)

    def toggle_label_visibility(self):
        if self.label_actor is None:
            return

        self.label_visible = not self.label_visible
        self.label_actor.SetVisibility(self.label_visible)
        self.plotter.render()
        self.toggle_label_button.setText("👁 隐藏标签" if self.label_visible else "👁 显示标签")

    def save_screenshot(self):
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        filename = f"screenshot_{timestamp}.png"
        self.plotter.screenshot(filename)
        self.statusBar().showMessage(f"已保存截图为 {filename}", 5000)

    def show_volume_stats(self, label_data, spacing):
        unique_labels = np.unique(label_data)
        voxel_volume = np.prod(spacing)
        text_lines = ["<b>体积统计（单位：mm³）</b>"]

        for label_val in unique_labels:
            if label_val == 0:
                continue  # 跳过背景
            voxel_count = np.sum(label_data == label_val)
            volume_mm3 = voxel_count * voxel_volume
            text_lines.append(f"标签 {int(label_val)}: {volume_mm3:.2f} mm³")

        self.volume_label.setText("<br>".join(text_lines))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
