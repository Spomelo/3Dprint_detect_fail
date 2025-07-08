import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from rknn_executor import RKNN_model_container

class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D打印缺陷检测系统 V1.0")
        self.root.geometry("1200x800")
        
        # 增强字体配置 - 使用更清晰的字体和大小
        self.font_style = ('Microsoft YaHei', 11)  # 增大字号
        self.title_font = ('Microsoft YaHei', 13, 'bold')  # 增大标题字号
        self.btn_font = ('Microsoft YaHei', 11, 'bold')  # 按钮专用粗体字
        
        # 初始化模型参数
        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.45
        self.IMG_SIZE = (640, 640)
        self.CLASSES = ('spaghetti', 'zits', 'stringing')
        
        # 模型和视频参数
        self.model = None
        self.cap = None
        self.detection_active = False
        self.video_paused = False
        self.current_model = tk.StringVar(value="best.rknn")
        self.platform = 'rknn'
        
        # 初始化UI
        self.create_widgets()
        self.init_model()

    def create_widgets(self):
        # 左侧控制面板 - 使用更深的背景色增加对比度
        left_frame = ttk.Frame(self.root, width=150, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 配置按钮样式 - 增强对比度
        style = ttk.Style()
        style.configure("TButton", 
                        font=self.btn_font,
                        padding=6,
                        background="#e0e0e0",  # 浅灰色背景
                        foreground="#000000")  # 黑色文字
        style.map("TButton", 
                 background=[('active', '#d0d0d0')])  # 活动状态更深的灰色

        # 功能按钮分组
        ttk.Label(left_frame, text="实时检测", font=self.title_font, 
                 foreground="#2c3e50").pack(pady=(15, 8), anchor=tk.W)
        self.add_button(left_frame, "打开摄像头", self.start_camera)
        self.add_button(left_frame, "停止检测", self.stop_detection)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        
        ttk.Label(left_frame, text="文件检测", font=self.title_font, 
                 foreground="#2c3e50").pack(pady=(15, 8), anchor=tk.W)
        self.add_button(left_frame, "打开图片", self.open_image)
        self.add_button(left_frame, "打开视频", self.open_video)
        self.pause_btn = self.add_button(left_frame, "暂停/继续", self.toggle_pause, state=tk.DISABLED)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        
        ttk.Label(left_frame, text="系统控制", font=self.title_font, 
                 foreground="#2c3e50").pack(pady=(15, 8), anchor=tk.W)
        self.add_button(left_frame, "退出程序", self.exit_app)

        # 右侧参数面板
        right_frame = ttk.Frame(self.root, width=230)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # 参数设置
        param_frame = ttk.LabelFrame(right_frame, text="检测参数",padding=10)
        param_frame.pack(pady=5, fill=tk.X)
        
        # 置信度调节
        self.conf_thresh = tk.DoubleVar(value=self.OBJ_THRESH)
        ttk.Label(param_frame, text="置信度阈值", font=self.font_style).pack(anchor=tk.W, pady=(0, 5))
        ttk.Scale(param_frame, from_=0, to=1, variable=self.conf_thresh,
                 orient=tk.HORIZONTAL, command=self.update_threshold).pack(fill=tk.X)
        
        # IOU调节
        self.iou_thresh = tk.DoubleVar(value=self.NMS_THRESH)
        ttk.Label(param_frame, text="IOU阈值", font=self.font_style).pack(anchor=tk.W, pady=(10, 5))
        ttk.Scale(param_frame, from_=0, to=1, variable=self.iou_thresh,
                 orient=tk.HORIZONTAL, command=self.update_threshold).pack(fill=tk.X)

        # 结果统计
        result_frame = ttk.LabelFrame(right_frame, text="检测结果", style="Red.TLabelframe", padding=10)
        result_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # 配置结果表格样式
        tree_style = ttk.Style()
        tree_style.configure("Treeview", 
                             font=self.font_style,
                             rowheight=28,
                             background="#ffffff",
                             fieldbackground="#ffffff")
        tree_style.configure("Treeview.Heading", 
                             font=('Microsoft YaHei', 11, 'bold'),
                             background="#eaeaea")

        self.tree = ttk.Treeview(
            result_frame,
            columns=("class", "count"),
            show="headings",
            style="Treeview"
        )

        # 配置列
        self.tree.column("class", width=50, anchor=tk.CENTER)
        self.tree.column("count", width=20, anchor=tk.CENTER)
        self.tree.heading("class", text="缺陷类型")
        self.tree.heading("count", text="数量")
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 主显示区域
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.main_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 文件路径显示
        self.file_label = ttk.Label(self.main_frame, text="当前文件: 无", font=self.font_style)
        self.file_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 5))

        # 状态栏 - 增强对比度
        self.status = ttk.Label(self.root, text="状态: 等待操作", 
                              relief=tk.SUNKEN, padding=5,
                              font=self.font_style,
                              background="#eaeaea",
                              foreground="#333333")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def add_button(self, parent, text, command, state=tk.NORMAL):
        """创建统一风格的按钮"""
        btn = ttk.Button(parent, text=text, command=command, state=state)
        btn.pack(pady=8, fill=tk.X, ipady=4)  # 增加垂直填充使按钮更高
        return btn

    def init_model(self):
        try:
            self.model = RKNN_model_container('./model/best.rknn', 'rk3588')
            self.status.config(text="状态: 模型加载成功")
        except Exception as e:
            self.status.config(text=f"状态: 模型加载失败 - {str(e)}")

    def update_image(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # 保持引用防止被垃圾回收

    def process_frame(self, frame):
        # 预处理
        img_src = cv2.resize(frame, self.IMG_SIZE)
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        
        # 模型推理
        outputs = self.model.run([img])  # type: ignore
        boxes, classes, scores = self.post_process(outputs)
        
        # 绘制检测结果
        if boxes is not None:
            self.draw(img_src, boxes, scores, classes)
        
        # 更新统计
        self.update_statistics(classes)
        return img_src

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self.OBJ_THRESH)
        scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes."""
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        import torch
        x = torch.tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y.numpy()

    def box_process(self, position):
        """将模型的输出转换为边界框坐标"""
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.IMG_SIZE[1] // grid_h, self.IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def post_process(self, input_data):
        """整合所有输出并进行后处理"""
        boxes, scores, classes_conf = [], [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # 根据阈值过滤
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # NMS
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def draw(self, image, boxes, scores, classes):
        """在图像上面绘制检测框与标签"""
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            print("%s @ (%d %d %d %d) %.3f" % (self.CLASSES[cl], top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def update_statistics(self, classes):
        counts = {}
        if classes is not None:
            for cls in classes:
                class_name = self.CLASSES[cls]
                counts[class_name] = counts.get(class_name, 0) + 1
        
        self.tree.delete(*self.tree.get_children())
        for cls, count in counts.items():
            self.tree.insert("", tk.END, values=(cls, count))

    def video_loop(self):
        while self.detection_active:
            if self.cap and not self.video_paused:
                ret, frame = self.cap.read()
                if ret:
                    processed_frame = self.process_frame(frame)
                    self.update_image(processed_frame)
                else:
                    # 视频结束
                    if hasattr(self, 'video_path'):
                        self.status.config(text=f"状态: 视频检测完成 - {os.path.basename(self.video_path)}")
                    self.stop_detection()
            self.root.update()

    def start_camera(self):
        self.stop_detection()
        self.cap = cv2.VideoCapture(21)  # 使用默认摄像头
        if not self.cap.isOpened():
            self.status.config(text="状态: 摄像头打开失败")
            return
        
        self.detection_active = True
        self.video_paused = False
        self.pause_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.video_loop, daemon=True).start()
        self.status.config(text="状态: 实时检测中...")
        self.file_label.config(text="当前文件: 摄像头实时画面")

    def open_image(self):
        self.stop_detection()
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("无法读取图片文件")
                
            processed_image = self.process_frame(image)
            self.update_image(processed_image)
            self.status.config(text=f"状态: 图片检测完成 - {os.path.basename(file_path)}")
            self.file_label.config(text=f"当前文件: {os.path.basename(file_path)}")
            
            # 显示原始图片尺寸信息
            height, width = image.shape[:2]
            self.status.config(text=f"状态: 图片检测完成 - {os.path.basename(file_path)} ({width}x{height})")
            
        except Exception as e:
            self.status.config(text=f"错误: {str(e)}")

    def open_video(self):
        self.stop_detection()
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if not file_path:
            return
        
        try:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            self.video_path = file_path
            self.detection_active = True
            self.video_paused = False
            self.pause_btn.config(state=tk.NORMAL)
            
            # 获取视频信息
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            self.status.config(text=f"状态: 视频检测中... - {os.path.basename(file_path)}")
            self.file_label.config(text=f"当前文件: {os.path.basename(file_path)} (FPS: {fps:.1f}, 时长: {duration:.1f}秒)")
            
            threading.Thread(target=self.video_loop, daemon=True).start()
            
        except Exception as e:
            self.status.config(text=f"错误: {str(e)}")

    def toggle_pause(self):
        self.video_paused = not self.video_paused
        if self.video_paused:
            self.status.config(text="状态: 视频已暂停")
        else:
            self.status.config(text="状态: 视频检测中...")

    def stop_detection(self):
        self.detection_active = False
        self.video_paused = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.pause_btn.config(state=tk.DISABLED)
        self.status.config(text="状态: 检测已停止")

    def update_threshold(self, event=None):
        self.OBJ_THRESH = self.conf_thresh.get()
        self.NMS_THRESH = self.iou_thresh.get()

    def exit_app(self):
        self.stop_detection()
        if self.model:
            self.model.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DefectDetectionApp(root)
    root.mainloop()