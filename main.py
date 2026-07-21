# -*- coding: utf-8 -*-
"""
سامانه جامع بینایی ماشین و پردازش تصویر
طراحی شده برای محیط ویندوز با استفاده از Tkinter و OpenCV و Matplotlib
نسخه کاملاً اصلاح شده و بدون باگ - منطبق بر مستندات گزارش کار
"""

import tkinter as cv_gui
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class AdvancedVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("computer vision")
        self.root.geometry("1300x850")
        
        # Maximize window on startup
        try:
            self.root.state('zoomed')
        except Exception:
            pass

        self.original_image = None
        self.processed_image = None
        self.display_image_ref = None
        self.history = []
            
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Persian-friendly fonts (Vazirmatn or Tahoma fallbacks)
        font_family = "Tahoma"
        style.configure("TButton", font=(font_family, 10, 'bold'), padding=6)
        style.configure("TLabel", font=(font_family, 10))
        style.configure("Header.TLabel", font=(font_family, 12, 'bold'), foreground="#10b981")
        style.configure("TCombobox", font=(font_family, 10))

        self.create_layout()
        self.load_default_pattern()

    def create_layout(self):
        # 1. Top Control Bar
        top_frame = ttk.Frame(self.root, padding=10, relief="raised")
        top_frame.pack(side="top", fill="x")

        ttk.Button(top_frame, text="بارگذاری تصویر 📂", command=self.load_image).pack(side="right", padx=5)
        ttk.Button(top_frame, text="ذخیره تصویر نهایی 💾", command=self.save_image).pack(side="right", padx=5)
        ttk.Button(top_frame, text="تثبیت گام (اعمال مجدد) ⚙️", command=self.commit_to_base).pack(side="right", padx=5)
        
        ttk.Button(top_frame, text="واگرد (Undo) ↩️", command=self.undo_action).pack(side="left", padx=5)
        ttk.Button(top_frame, text="ریست کامل 🔄", command=self.reset_image).pack(side="left", padx=5)
        ttk.Button(top_frame, text="نمایش هیستوگرام 📊", command=self.show_histogram).pack(side="left", padx=5)
        ttk.Button(top_frame, text="الگوهای تستی 💠", command=self.show_patterns_dialog).pack(side="left", padx=5)

        # 2. Main Paned Layout
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left panel: control sidebar
        control_frame = ttk.Frame(main_paned, width=320, relief="groove", padding=12)
        main_paned.add(control_frame, weight=1)

        ttk.Label(control_frame, text="تنظیمات الگوریتم‌ها", style="Header.TLabel").pack(pady=(0, 15))

        # Category Selector
        ttk.Label(control_frame, text="انتخاب فصل آموزشی:").pack(anchor="ne", pady=(5, 2))
        self.category_var = cv_gui.StringVar()
        self.categories = {
            "بخش اول: مبانی، درونیابی و اشکال": [
                "Grayscale",
                "Nearest Neighbor Resizing",
                "Bilinear Resizing",
                "Bicubic Resizing",
                "Euclidean Distance",
                "City Block Distance (D4)",
                "Chessboard Distance (D8)",
                "Find Contours", 
                "Hough Line Transform"
            ],
            "بخش دوم: بهبود در حوزه مکان": [
                "Negative", 
                "Log Transformation", 
                "Gamma Correction", 
                "Histogram Equalization",
                "Histogram Matching",
                "Local Histogram Equalization (CLAHE)",
                "Gaussian Blur", 
                "Median Blur", 
                "Average Blur", 
                "Bilateral Filter",
                "Laplacian", 
                "Sobel X", 
                "Sobel Y", 
                "Sobel Combined",
                "Robert Cross Edge",
                "Canny Edge Detector",
                "Fuzzy Contrast Enhancement"
            ],
            "بخش سوم: فیلترینگ در حوزه فرکانس": [
                "FFT Spectrum", 
                "Ideal LowPass", 
                "Ideal HighPass",
                "Butterworth Lowpass",
                "Butterworth Highpass",
                "Gaussian Lowpass",
                "Gaussian Highpass",
                "High-frequency Emphasis",
                "Homomorphic Filtering",
                "Ideal Bandpass",
                "Ideal Bandreject",
                "Butterworth Bandpass",
                "Butterworth Bandreject",
                "Ideal Notch Reject",
                "Butterworth Notch Reject",
                "Gaussian Notch Reject"
            ],
            "بخش چهارم: ترمیم، قطعه‌بندی و رنگ": [
                "Add Salt&Pepper Noise", 
                "Add Gaussian Noise",
                "Geometric Mean Filter",
                "Harmonic Mean Filter",
                "Contraharmonic Mean Filter",
                "Max Filter",
                "Min Filter",
                "Midpoint Filter",
                "Alpha-trimmed Mean Filter",
                "Adaptive Local Noise Reduction",
                "Adaptive Median Filter",
                "Erosion", 
                "Dilation", 
                "Opening", 
                "Closing", 
                "Morphological Gradient",
                "Global Thresholding", 
                "Adaptive Mean Threshold", 
                "Adaptive Gaussian Threshold", 
                "Otsu Thresholding",
                "RGB to HSV", 
                "RGB to LAB", 
                "RGB to YCrCb"
            ]
        }
        self.combo_category = ttk.Combobox(control_frame, textvariable=self.category_var, values=list(self.categories.keys()), state="readonly")
        self.combo_category.pack(fill="x", pady=5)
        self.combo_category.bind("<<ComboboxSelected>>", self.update_algorithm_list)

        # Algorithm Selector
        ttk.Label(control_frame, text="انتخاب الگوریتم:").pack(anchor="ne", pady=(10, 2))
        self.algo_var = cv_gui.StringVar()
        self.combo_algo = ttk.Combobox(control_frame, textvariable=self.algo_var, state="readonly")
        self.combo_algo.pack(fill="x", pady=5)
        self.combo_algo.bind("<<ComboboxSelected>>", self.reset_params)

        # Separator
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=15)

        # Dynamic Sliders & Adjustments
        self.lbl_param1 = ttk.Label(control_frame, text="پارامتر 1:")
        self.lbl_param1.pack(anchor="ne", pady=(5, 0))
        self.slider_param1 = ttk.Scale(control_frame, from_=1, to=100, orient="horizontal")
        self.slider_param1.pack(fill="x", pady=2)
        
        self.lbl_param2 = ttk.Label(control_frame, text="پارامتر 2:")
        self.lbl_param2.pack(anchor="ne", pady=(10, 0))
        self.slider_param2 = ttk.Scale(control_frame, from_=1, to=100, orient="horizontal")
        self.slider_param2.pack(fill="x", pady=2)

        # Run Button
        ttk.Button(control_frame, text="اعمال فیلتر/الگوریتم ▶️", command=self.apply_algorithm).pack(fill="x", pady=25)
        
        # Stats info panel
        info_group = ttk.LabelFrame(control_frame, text=" 📊 مشخصات تصویر ", padding=10)
        info_group.pack(side="bottom", fill="x", pady=10)
        
        self.info_label = ttk.Label(info_group, text="در انتظار بارگذاری تصویر...", justify="right", wraplength=280)
        self.info_label.pack(fill="x")

        # Right panel: visual viewport area
        display_frame = ttk.Frame(main_paned, relief="sunken", padding=5)
        main_paned.add(display_frame, weight=4)

        # Split visual area (Original vs Processed side-by-side)
        self.visuals_pane = ttk.PanedWindow(display_frame, orient="horizontal")
        self.visuals_pane.pack(fill="both", expand=True)

        # Original image canvas panel
        left_canvas_frame = ttk.LabelFrame(self.visuals_pane, text=" تصویر اصلی ورودی ")
        self.canvas_original = cv_gui.Canvas(left_canvas_frame, bg="#212121", highlightthickness=0)
        self.canvas_original.pack(fill="both", expand=True, padx=2, pady=2)
        self.visuals_pane.add(left_canvas_frame, weight=1)

        # Processed image canvas panel
        right_canvas_frame = ttk.LabelFrame(self.visuals_pane, text=" نتیجه پردازش تصویر ")
        self.canvas_processed = cv_gui.Canvas(right_canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self.canvas_processed.pack(fill="both", expand=True, padx=2, pady=2)
        self.visuals_pane.add(right_canvas_frame, weight=1)

        # Set default values
        self.combo_category.current(0)
        self.update_algorithm_list(None)

    def update_algorithm_list(self, event):
        selected_cat = self.category_var.get()
        self.combo_algo['values'] = self.categories[selected_cat]
        self.combo_algo.current(0)
        self.reset_params(None)

    def reset_params(self, event):
        algo = self.algo_var.get()
        
        # Reset slider states
        self.slider_param1.state(['!disabled'])
        self.slider_param2.state(['!disabled'])
        self.lbl_param1.config(text="پارامتر 1:")
        self.lbl_param2.config(text="پارامتر 2:")

        if "Resizing" in algo:
            self.slider_param1.config(from_=10, to=300, value=150)
            self.lbl_param1.config(text="درصد مقیاس دهی (%):")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Distance" in algo:
            self.slider_param1.config(from_=0, to=255, value=127)
            self.lbl_param1.config(text="آستانه باینری سازی تصویر:")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif algo == "Histogram Matching":
            self.slider_param1.config(from_=10, to=240, value=120)
            self.lbl_param1.config(text="میانگین هدف توزیع:")
            self.slider_param2.config(from_=5, to=100, value=30)
            self.lbl_param2.config(text="انحراف معیار هدف توزیع:")

        elif "CLAHE" in algo:
            self.slider_param1.config(from_=2, to=32, value=8)
            self.lbl_param1.config(text="اندازه مش شبکه محلی:")
            self.slider_param2.config(from_=5, to=100, value=20)
            self.lbl_param2.config(text="کنتراست آستانه (Clip/10):")

        elif algo == "High-boost Filtering":
            self.slider_param1.config(from_=10, to=50, value=15)
            self.lbl_param1.config(text="ضریب تقویت (A/10):")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Butterworth" in algo and "Notch" not in algo:
            self.slider_param1.config(from_=5, to=200, value=40)
            self.lbl_param1.config(text="شعاع فرکانس قطع (D0):")
            self.slider_param2.config(from_=1, to=10, value=2)
            self.lbl_param2.config(text="درجه فیلتر باترورث (n):")

        elif "Gaussian" in algo and "Notch" not in algo:
            self.slider_param1.config(from_=5, to=200, value=40)
            self.lbl_param1.config(text="شعاع فرکانس قطع (D0):")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif algo == "High-frequency Emphasis":
            self.slider_param1.config(from_=5, to=200, value=40)
            self.lbl_param1.config(text="شعاع فرکانس قطع (D0):")
            self.slider_param2.config(from_=5, to=50, value=15)
            self.lbl_param2.config(text="ضریب تقویت فرکانس بالا (k/10):")

        elif algo == "Homomorphic Filtering":
            self.slider_param1.config(from_=5, to=200, value=40)
            self.lbl_param1.config(text="فرکانس قطع (D0):")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Bandpass" in algo or "Bandreject" in algo:
            self.slider_param1.config(from_=5, to=200, value=50)
            self.lbl_param1.config(text="فرکانس مرکزی (D0):")
            self.slider_param2.config(from_=2, to=100, value=15)
            self.lbl_param2.config(text="پهنای باند فرکانسی (W):")

        elif "Notch" in algo:
            self.slider_param1.config(from_=2, to=50, value=15)
            self.lbl_param1.config(text="شعاع فیلتر شکاف:")
            if "Butterworth" in algo:
                self.slider_param2.config(from_=1, to=10, value=2)
                self.lbl_param2.config(text="درجه فیلتر (n):")
            else:
                self.slider_param2.state(['disabled'])
                self.lbl_param2.config(text="غیرفعال")

        elif "Contraharmonic" in algo:
            self.slider_param1.config(from_=3, to=15, value=5)
            self.lbl_param1.config(text="اندازه هسته (فقط فرد):")
            self.slider_param2.config(from_=-50, to=50, value=15)
            self.lbl_param2.config(text="مرتبه فیلتر (Q/10):")

        elif "Alpha-trimmed" in algo:
            self.slider_param1.config(from_=3, to=15, value=5)
            self.lbl_param1.config(text="اندازه همسایگی (فقط فرد):")
            self.slider_param2.config(from_=0, to=14, value=2)
            self.lbl_param2.config(text="تعداد حذفیات نویز (d):")

        elif algo == "Adaptive Local Noise Reduction":
            self.slider_param1.config(from_=3, to=15, value=5)
            self.lbl_param1.config(text="اندازه پنجره محلی (فقط فرد):")
            self.slider_param2.config(from_=10, to=1000, value=100)
            self.lbl_param2.config(text="واریانس نویز کل تصویر:")

        elif algo == "Adaptive Median Filter":
            self.slider_param1.config(from_=3, to=21, value=7)
            self.lbl_param1.config(text="حداکثر اندازه پنجره:")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Mean Filter" in algo or "Max Filter" in algo or "Min Filter" in algo or "Midpoint Filter" in algo:
            self.slider_param1.config(from_=3, to=15, value=5)
            self.lbl_param1.config(text="اندازه پنجره (فقط فرد):")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Blur" in algo:
            self.slider_param1.config(from_=3, to=25, value=5)
            self.lbl_param1.config(text="اندازه هسته (فقط فرد):")
            if algo == "Gaussian Blur":
                self.slider_param2.config(from_=1, to=100, value=15)
                self.lbl_param2.config(text="انحراف معیار گاوسی (Sigma/10):")
            else:
                self.slider_param2.state(['disabled'])
                self.lbl_param2.config(text="غیرفعال")

        elif "Bilateral" in algo:
            self.slider_param1.config(from_=5, to=50, value=15)
            self.lbl_param1.config(text="فاصله مکانی (Sigma S):")
            self.slider_param2.config(from_=5, to=100, value=25)
            self.lbl_param2.config(text="تفاوت رنگ (Sigma R):")

        elif "Canny" in algo:
            self.slider_param1.config(from_=1, to=150, value=20)
            self.lbl_param1.config(text="آستانه پایین (Low Threshold):")
            self.slider_param2.config(from_=10, to=250, value=50)
            self.lbl_param2.config(text="آستانه بالا (High Threshold):")

        elif "Threshold" in algo:
            if "Adaptive" in algo:
                self.slider_param1.config(from_=3, to=45, value=11)
                self.lbl_param1.config(text="اندازه بلوک همسایگی:")
                self.slider_param2.config(from_=-20, to=40, value=5)
                self.lbl_param2.config(text="ثابت کاهشی (C):")
            else:
                self.slider_param1.config(from_=0, to=255, value=127)
                self.lbl_param1.config(text="مقدار آستانه مرزی:")
                self.slider_param2.state(['disabled'])
                self.lbl_param2.config(text="غیرفعال")

        elif "Gamma" in algo:
            self.slider_param1.config(from_=1, to=50, value=10)
            self.lbl_param1.config(text="مقدار گاما (تقسیم بر ۱۰):")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Ideal LowPass" in algo or "Ideal HighPass" in algo:
            self.slider_param1.config(from_=5, to=60, value=20)
            self.lbl_param1.config(text="شعاع فرکانسی قطع (D0):")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Noise" in algo:
            if "Salt&Pepper" in algo:
                self.slider_param1.config(from_=1, to=40, value=5)
                self.lbl_param1.config(text="درصد نویز (%):")
                self.slider_param2.state(['disabled'])
                self.lbl_param2.config(text="غیرفعال")
            else:
                self.slider_param1.config(from_=5, to=100, value=25)
                self.lbl_param1.config(text="انحراف معیار نویز (Sigma):")
                self.slider_param2.state(['disabled'])
                self.lbl_param2.config(text="غیرفعال")

        elif algo in ["Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"]:
            self.slider_param1.config(from_=3, to=15, value=5)
            self.lbl_param1.config(text="اندازه المان ساختاری:")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        elif "Hough" in algo:
            self.slider_param1.config(from_=20, to=150, value=50)
            self.lbl_param1.config(text="آستانه انباشتگر هاف:")
            self.slider_param2.state(['disabled'])
            self.lbl_param2.config(text="غیرفعال")

        else:
            self.slider_param1.state(['disabled'])
            self.slider_param2.state(['disabled'])
            self.lbl_param1.config(text="پارامتر بلااستفاده")
            self.lbl_param2.config(text="پارامتر بلااستفاده")

    def load_default_pattern(self):
        # Generate elegant geometric shape pattern as default
        pattern = np.zeros((400, 400, 3), dtype=np.uint8)
        # Background split
        pattern[:, :200] = [240, 240, 240]
        pattern[:, 200:] = [40, 40, 40]
        # Circle
        cv2.circle(pattern, (200, 200), 80, (0, 240, 0), -1)
        cv2.circle(pattern, (200, 200), 80, (0, 0, 0), 3)
        # Rectangle
        cv2.rectangle(pattern, (60, 160), (140, 240), (240, 150, 0), -1)
        cv2.rectangle(pattern, (60, 160), (140, 240), (0, 0, 0), 2)
        # Text
        cv2.putText(pattern, "CV LAB 2026", (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        
        self.original_image = pattern.copy()
        self.processed_image = pattern.copy()
        self.history = []
        self.show_images()
        self.update_info(self.processed_image)

    def show_patterns_dialog(self):
        # Allow user to pick from multiple academic test patterns
        dialog = cv_gui.Toplevel(self.root)
        dialog.title("انتخاب الگوی تست استاندارد")
        dialog.geometry("350x250")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="یک الگوی ریاضی انتخاب کنید:", font=("Tahoma", 10, "bold")).pack(pady=15)

        def set_pattern(p_type):
            if p_type == 'stripes':
                # Grid patterns / lines
                pattern = np.zeros((400, 400, 3), dtype=np.uint8)
                for x in range(0, 400, 15):
                    cv2.line(pattern, (x, 0), (x, 400), (255, 255, 255), 2)
                    cv2.line(pattern, (0, x), (400, x), (180, 180, 180), 1)
                self.original_image = pattern
            elif p_type == 'coins':
                # Smooth circles simulating grey coins with salt and pepper
                pattern = np.full((400, 400, 3), 45, dtype=np.uint8)
                # Draw coins with different intensities
                cv2.circle(pattern, (100, 120), 40, (180, 180, 180), -1)
                cv2.circle(pattern, (280, 100), 45, (220, 220, 220), -1)
                cv2.circle(pattern, (140, 280), 35, (140, 140, 140), -1)
                cv2.circle(pattern, (270, 270), 50, (200, 200, 200), -1)
                # Inject intense salt and pepper noise
                noise = np.random.rand(400, 400)
                pattern[noise < 0.03] = [0, 0, 0]
                pattern[noise > 0.97] = [255, 255, 255]
                self.original_image = pattern
            elif p_type == 'gradient':
                # Grayscale gradient
                pattern = np.zeros((400, 400, 3), dtype=np.uint8)
                for y in range(400):
                    val = int((y / 400.0) * 255)
                    pattern[y, :] = [val, val, val]
                self.original_image = pattern
            
            self.processed_image = self.original_image.copy()
            self.history = []
            self.show_images()
            self.update_info(self.processed_image)
            dialog.destroy()

        ttk.Button(dialog, text="شبکه فرکانسی و تاروپود خطوط", command=lambda: set_pattern('stripes')).pack(fill="x", padx=30, pady=5)
        ttk.Button(dialog, text="سکه‌های سیمانی نویزی (حذف نویز/قطعه‌بندی)", command=lambda: set_pattern('coins')).pack(fill="x", padx=30, pady=5)
        ttk.Button(dialog, text="گرادیان تدریجی روشنایی", command=lambda: set_pattern('gradient')).pack(fill="x", padx=30, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.bmp;*.jpeg;*.tif;*.tiff")])
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("خطا", "تصویر بارگذاری نشد.")
                return
            
            self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.processed_image = self.original_image.copy()
            self.history = []
            self.show_images()
            self.update_info(self.processed_image)

    def show_images(self):
        self.display_on_canvas(self.canvas_original, self.original_image)
        self.display_on_canvas(self.canvas_processed, self.processed_image)

    def display_on_canvas(self, canvas, img_array):
        if img_array is None: return
        
        # Force canvas to update size
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 500
            canvas_height = 450

        pil_image = Image.fromarray(img_array)
        
        img_width, img_height = pil_image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (max(10, int(img_width*ratio)), max(10, int(img_height*ratio)))
        
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Keep references to avoid garbage collection bug in Tkinter
        photo = ImageTk.PhotoImage(pil_image)
        if canvas == self.canvas_original:
            self.display_orig_ref = photo
        else:
            self.display_proc_ref = photo
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor="center")

    def save_image(self):
        if self.processed_image is None: return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            save_img = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, save_img)
            messagebox.showinfo("موفق", "تصویر با موفقیت ذخیره شد.")

    def commit_to_base(self):
        """تثبیت گام فعلی جهت اعمال عملیاتهای زنجیره‌ای"""
        if self.processed_image is not None:
            self.history.append(self.original_image.copy())
            self.original_image = self.processed_image.copy()
            self.show_images()
            messagebox.showinfo("تثبیت", "نتیجه به عنوان تصویر پایه تثبیت شد. اکنون می‌توانید الگوریتم بعدی را اعمال کنید.")

    def undo_action(self):
        if self.history:
            self.processed_image = self.original_image.copy()
            self.original_image = self.history.pop()
            self.show_images()
            self.update_info(self.processed_image)
        else:
            messagebox.showwarning("هشدار", "تاریخچه خالی است و گام دورتری وجود ندارد.")

    def reset_image(self):
        if self.original_image is not None:
            self.load_default_pattern()

    def update_info(self, img):
        if img is None: return
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        dtype = img.dtype
        mean_intensity = np.mean(img)
        std_dev = np.std(img)
        
        stats = (f"ابعاد تصویر: {w}x{h} پیکسل\n"
                 f"تعداد کانال‌ها: {c}\n"
                 f"فرمت داده: {dtype}\n"
                 f"میانگین روشنایی: {mean_intensity:.2f}\n"
                 f"انحراف معیار: {std_dev:.2f}")
        self.info_label.config(text=stats)

    def apply_algorithm(self):
        if self.original_image is None:
            messagebox.showwarning("خطا", "لطفاً ابتدا تصویری بارگذاری یا الگو انتخاب کنید.")
            return

        algo = self.algo_var.get()
        p1 = int(self.slider_param1.get())
        
        # Handle param2 properly if it's disabled
        p2 = 0
        try:
            if str(self.slider_param2.cget("state")) != "disabled":
                p2 = int(self.slider_param2.get())
        except Exception:
            p2 = int(self.slider_param2.get())

        self.history.append(self.original_image.copy())

        try:
            img = self.original_image.copy()
            
            # Grayscale buffer
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img.copy()

            # ---------------------------------------------
            # بخش اول: مبانی، درونیابی و اشکال
            # ---------------------------------------------
            if algo == "Grayscale":
                # تبدیل به تصویر سطوح خاکستری استاندارد
                res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            elif algo == "Nearest Neighbor Resizing":
                # درونیابی پیکسل نزدیکترین همسایه برای تغییر اندازه
                scale_percent = p1
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

            elif algo == "Bilinear Resizing":
                # درونیابی پیکسل دوخطی (Bilinear) برای تغییر اندازه
                scale_percent = p1
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

            elif algo == "Bicubic Resizing":
                # درونیابی پیکسل دومکعبی (Bicubic) برای تغییر اندازه
                scale_percent = p1
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

            elif algo == "Euclidean Distance":
                # محاسبه تبدیل فاصله اقلیدسی
                _, thresh = cv2.threshold(gray, p1, 255, cv2.THRESH_BINARY)
                dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
                dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img = cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2RGB)

            elif algo == "City Block Distance (D4)":
                # محاسبه تبدیل فاصله بلوک شهری یا D4
                _, thresh = cv2.threshold(gray, p1, 255, cv2.THRESH_BINARY)
                dist = cv2.distanceTransform(thresh, cv2.DIST_L1, 3)
                dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img = cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2RGB)

            elif algo == "Chessboard Distance (D8)":
                # محاسبه تبدیل فاصله شطرنجی یا D8
                _, thresh = cv2.threshold(gray, p1, 255, cv2.THRESH_BINARY)
                dist = cv2.distanceTransform(thresh, cv2.DIST_C, 3)
                dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img = cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2RGB)

            elif algo == "Find Contours":
                # استخراج و رسم کانتورها (مرز اشکال)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            
            elif algo == "Hough Line Transform":
                # تبدیل هاف برای شناسایی خطوط مستطیلی و مورب در تصویر
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                h_thresh = p1 if p1 > 10 else 50
                lines = cv2.HoughLines(edges, 1, np.pi/180, h_thresh)
                img_lines = img.copy()
                if lines is not None:
                    for line in lines[:15]:
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
                img = img_lines

            # ---------------------------------------------
            # بخش دوم: بهبود در حوزه مکان
            # ---------------------------------------------
            elif algo == "Negative":
                # نگاتیو کردن تصویر (معکوس‌سازی شدت روشنایی)
                img = 255 - img

            elif algo == "Log Transformation":
                # نگاشت لگاریتمی برای افزایش کنتراست جزئیات تیره
                max_val = np.max(img)
                if max_val == 0:
                    max_val = 1
                c = 255 / np.log(1 + max_val)
                log_image = c * (np.log(img.astype(np.float32) + 1))
                img = np.array(log_image, dtype=np.uint8)

            elif algo == "Gamma Correction":
                # تصحیح گاما برای تنظیم توان غیرخطی شدت نور
                gamma = p1 / 10.0 if p1 > 0 else 0.1
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                img = cv2.LUT(img, table)

            elif algo == "Histogram Equalization":
                # همسان‌سازی سراسری هیستوگرام
                if len(img.shape) == 3:
                    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
                else:
                    img = cv2.equalizeHist(img)

            elif algo == "Histogram Matching":
                # تطبیق هیستوگرام با توزیع هدف گاوسی تعیین شده توسط کاربر
                hist_in, _ = np.histogram(gray.flatten(), 256, [0, 256])
                cdf_in = hist_in.cumsum()
                cdf_in_norm = cdf_in * 255 / (cdf_in[-1] if cdf_in[-1] > 0 else 1)

                mean_target = p1
                std_target = p2 if p2 > 0 else 30
                target_pdf = np.exp(-((np.arange(256) - mean_target) ** 2) / (2 * (std_target ** 2)))
                target_pdf /= (target_pdf.sum() if target_pdf.sum() > 0 else 1)
                cdf_target = target_pdf.cumsum()
                cdf_target_norm = cdf_target * 255 / (cdf_target[-1] if cdf_target[-1] > 0 else 1)

                lut = np.zeros(256, dtype=np.uint8)
                g_idx = 0
                for i in range(256):
                    while g_idx < 255 and cdf_target_norm[g_idx] < cdf_in_norm[i]:
                        g_idx += 1
                    lut[i] = g_idx

                res = cv2.LUT(gray, lut)
                img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            elif algo == "Local Histogram Equalization (CLAHE)":
                # همسان‌سازی محلی هیستوگرام با محدودیت کنتراست
                grid_size = p1 if p1 > 1 else 8
                clip_limit = p2 / 10.0 if p2 > 0 else 2.0
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                if len(img.shape) == 3:
                    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
                    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
                else:
                    img = clahe.apply(img)

            elif algo == "Gaussian Blur":
                # فیلتر هموارکننده گاوسی مکانی
                k = p1 if p1 % 2 == 1 else p1 + 1
                sigma = p2 / 10.0 if p2 > 0 else 1.5
                img = cv2.GaussianBlur(img, (k, k), sigma)

            elif algo == "Median Blur":
                # فیلتر میانه مکانی برای حذف نویز نمک و فلفل
                k = p1 if p1 % 2 == 1 else p1 + 1
                img = cv2.medianBlur(img, k)

            elif algo == "Average Blur":
                # فیلتر میانگین مکانی (باکس فیلتر)
                k = p1 if p1 % 2 == 1 else p1 + 1
                img = cv2.blur(img, (k, k))
            
            elif algo == "Bilateral Filter":
                # فیلتر دوجانبه برای هموارسازی با حفظ لبه‌های تصویر
                img = cv2.bilateralFilter(img, 9, p2, p1)

            elif algo == "Laplacian":
                # مشتق مرتبه دوم لاپلاسین برای تیزکردن و استخراج لبه
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                img = cv2.convertScaleAbs(laplacian)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Sobel X":
                # فیلتر سوبل افقی (مشتق جزئی نسبت به X)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                abs_sobel = np.absolute(sobelx)
                img = np.uint8(abs_sobel)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Sobel Y":
                # فیلتر سوبل عمودی (مشتق جزئی نسبت به Y)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                abs_sobel = np.absolute(sobely)
                img = np.uint8(abs_sobel)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Sobel Combined":
                # ترکیب فیلترهای سوبل افقی و عمودی
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                abs_x = np.absolute(sobelx)
                abs_y = np.absolute(sobely)
                combined = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
                img = np.uint8(combined)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Robert Cross Edge":
                # عملگر لبه‌یاب روبرت بر اساس تفاضل قطری
                kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
                kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
                rx = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
                ry = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
                magnitude = np.sqrt(rx**2 + ry**2)
                res = cv2.convertScaleAbs(magnitude)
                img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            elif algo == "Canny Edge Detector":
                # لبه‌یاب کنی چند مرحله‌ای پیشرفته
                edges = cv2.Canny(gray, p1, p2)
                img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            elif algo == "High-boost Filtering":
                # فیلتر تقویت بالا (High-boost) بر اساس تفاضل از فیلتر بلور
                k_val = p1 / 10.0 if p1 > 0 else 1.5
                blur = cv2.GaussianBlur(img, (5, 5), 2.0)
                mask = img.astype(np.float32) - blur.astype(np.float32)
                boosted = img.astype(np.float32) + k_val * mask
                img = np.clip(boosted, 0, 255).astype(np.uint8)

            elif algo == "Fuzzy Contrast Enhancement":
                # پیاده‌سازی فیلترینگ و کنتراست فازی با تابع عضویت S-curve
                r_min, r_max = float(np.min(gray)), float(np.max(gray))
                if r_max > r_min:
                    mu = (gray.astype(np.float32) - r_min) / (r_max - r_min)
                    mu_prime = np.where(mu <= 0.5, 2 * (mu ** 2), 1 - 2 * ((1 - mu) ** 2))
                    enhanced = r_min + mu_prime * (r_max - r_min)
                    res = np.clip(enhanced, 0, 255).astype(np.uint8)
                    img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            # ---------------------------------------------
            # بخش سوم: فیلترهای فرکانسی
            # ---------------------------------------------
            elif algo == "FFT Spectrum":
                # نمایش طیف دامنه فرکانسی تبدیل فوریه دو بعدی تصویر
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
                magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)

            elif algo == "Ideal LowPass":
                # فیلتر فرکانسی پایین‌گذر ایده‌آل
                d0 = p1
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                mask = np.zeros((rows, cols), np.uint8)
                cv2.circle(mask, (ccol, crow), d0, 1, -1)
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Ideal HighPass":
                # فیلتر فرکانسی بالاگذر ایده‌آل
                d0 = p1
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                mask = np.ones((rows, cols), np.uint8)
                cv2.circle(mask, (ccol, crow), d0, 0, -1)
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * mask
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Butterworth Lowpass":
                # فیلتر فرکانسی پایین‌گذر باترورث
                d0 = p1
                n = p2 if p2 > 0 else 2
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                eps = 1e-8
                h_filter = 1 / (1 + (d / (d0 + eps))**(2 * n))
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Butterworth Highpass":
                # فیلتر فرکانسی بالاگذر باترورث
                d0 = p1
                n = p2 if p2 > 0 else 2
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                eps = 1e-8
                h_filter = 1 / (1 + ((d0 / (d + eps))**(2 * n)))
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Gaussian Lowpass":
                # فیلتر فرکانسی پایین‌گذر گوسی
                d0 = p1
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                h_filter = np.exp(-(d**2) / (2 * (d0**2) + 1e-8))
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Gaussian Highpass":
                # فیلتر فرکانسی بالاگذر گوسی
                d0 = p1
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                h_filter = 1 - np.exp(-(d**2) / (2 * (d0**2) + 1e-8))
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "High-frequency Emphasis":
                # فیلتر تاکید فرکانس‌های بالا
                d0 = p1
                k_val = p2 / 10.0 if p2 > 0 else 1.5
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                h_hp = 1 - np.exp(-(d**2) / (2 * (d0**2) + 1e-8))
                h_filter = 0.5 + k_val * h_hp
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Homomorphic Filtering":
                # فیلترینگ همومورفیک جهت تصحیح نور غیریکنواخت تصویر
                d0 = p1
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                img_log = np.log1p(gray.astype(np.float32))
                f_log = np.fft.fft2(img_log)
                fshift_log = np.fft.fftshift(f_log)
                gamma_L = 0.25
                gamma_H = 1.5
                h_filter = (gamma_H - gamma_L) * (1 - np.exp(-(d**2) / (2 * (d0**2) + 1e-8))) + gamma_L
                fshift_filtered = fshift_log * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img_exp = np.expm1(img_back)
                img = np.uint8(cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Ideal Bandpass":
                # فیلتر فرکانسی میان‌گذر ایده‌آل
                d0 = p1
                w_band = p2 if p2 > 0 else 15
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                h_filter = np.zeros_like(d)
                h_filter[(d >= d0 - w_band/2) & (d <= d0 + w_band/2)] = 1
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Ideal Bandreject":
                # فیلتر فرکانسی میان‌نگذر ایده‌آل
                d0 = p1
                w_band = p2 if p2 > 0 else 15
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                h_filter = np.ones_like(d)
                h_filter[(d >= d0 - w_band/2) & (d <= d0 + w_band/2)] = 0
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Butterworth Bandpass":
                # فیلتر فرکانسی میان‌گذر باترورث
                d0 = p1
                w_band = p2 if p2 > 0 else 15
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                denom = d**2 - d0**2
                denom[denom == 0] = 1e-8
                h_br = 1 / (1 + ((d * w_band) / denom)**(2 * 2))
                h_filter = 1 - h_br
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Butterworth Bandreject":
                # فیلتر فرکانسی میان‌نگذر باترورث
                d0 = p1
                w_band = p2 if p2 > 0 else 15
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d = np.sqrt(U**2 + V**2)
                denom = d**2 - d0**2
                denom[denom == 0] = 1e-8
                h_filter = 1 / (1 + ((d * w_band) / denom)**(2 * 2))
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo in ["Ideal Notch Reject", "Butterworth Notch Reject", "Gaussian Notch Reject"]:
                # فیلتر حذف نویز و شکاف یا هماهنگ کننده خودکار (Notch Filter)
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                mag = np.abs(fshift)
                mask_center = np.ones_like(mag)
                cv2.circle(mask_center, (ccol, crow), 15, 0, -1)
                masked_mag = mag * mask_center
                peak_y, peak_x = np.unravel_index(np.argmax(masked_mag), mag.shape)
                u0, v0 = peak_x - ccol, peak_y - crow
                u = np.arange(cols) - ccol
                v = np.arange(rows) - crow
                U, V = np.meshgrid(u, v)
                d1 = np.sqrt((U - u0)**2 + (V - v0)**2)
                d2 = np.sqrt((U + u0)**2 + (V + v0)**2)
                d0 = p1 if p1 > 0 else 15
                if algo == "Ideal Notch Reject":
                    h_filter = np.ones_like(d1)
                    h_filter[d1 <= d0] = 0
                    h_filter[d2 <= d0] = 0
                elif algo == "Butterworth Notch Reject":
                    n = p2 if p2 > 0 else 2
                    h_filter = 1 / (1 + (d0**2 / (d1 * d2 + 1e-8))**n)
                elif algo == "Gaussian Notch Reject":
                    h_filter = (1 - np.exp(-(d1**2) / (2 * (d0**2) + 1e-8))) * (1 - np.exp(-(d2**2) / (2 * (d0**2) + 1e-8)))
                fshift_filtered = fshift * h_filter
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # ---------------------------------------------
            # بخش چهارم: ترمیم، قطعه‌بندی و رنگ
            # ---------------------------------------------
            elif algo == "Add Salt&Pepper Noise":
                # اعمال نویز نمک و فلفل بر روی تصویر ورودی
                prob = p1 / 100.0
                output = np.copy(img)
                h, w = output.shape[:2]
                num_salt = int(prob * h * w * 0.5)
                num_pepper = int(prob * h * w * 0.5)
                if num_salt > 0:
                    coords_y = np.random.randint(0, h, num_salt)
                    coords_x = np.random.randint(0, w, num_salt)
                    output[coords_y, coords_x] = [255, 255, 255] if len(output.shape) == 3 else 255
                if num_pepper > 0:
                    coords_y = np.random.randint(0, h, num_pepper)
                    coords_x = np.random.randint(0, w, num_pepper)
                    output[coords_y, coords_x] = [0, 0, 0] if len(output.shape) == 3 else 0
                img = output

            elif algo == "Add Gaussian Noise":
                # اعمال نویز با توزیع مستقل گاوسی بر تصویر
                sigma = p1
                gauss = np.random.normal(0, sigma, img.shape).astype('float32')
                noisy = img.astype('float32') + gauss
                img = np.clip(noisy, 0, 255).astype(np.uint8)

            elif algo == "Geometric Mean Filter":
                # فیلتر ترمیم میانگین هندسی برای کاهش نویز نرم و تصادفی
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                log_img = np.log(img.astype(np.float32) + 1.0)
                mean_log = cv2.blur(log_img, (k_size, k_size))
                img = np.clip(np.exp(mean_log) - 1.0, 0, 255).astype(np.uint8)

            elif algo == "Harmonic Mean Filter":
                # فیلتر ترمیم میانگین هارمونیک مناسب برای نویز نمک
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                inv_img = 1.0 / (img.astype(np.float32) + 1e-8)
                sum_inv = cv2.blur(inv_img, (k_size, k_size))
                img = np.clip(1.0 / (sum_inv + 1e-8), 0, 255).astype(np.uint8)

            elif algo == "Contraharmonic Mean Filter":
                # فیلتر ترمیم میانگین کنتراهارمونیک مناسب نویز فلفل (مرتبه مثبت) یا نمک (مرتبه منفی)
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                q_val = p2 / 10.0
                img_f = img.astype(np.float32) + 1e-8
                num = cv2.blur(img_f ** (q_val + 1), (k_size, k_size)) * (k_size * k_size)
                denom = cv2.blur(img_f ** q_val, (k_size, k_size)) * (k_size * k_size)
                img = np.clip(num / (denom + 1e-8), 0, 255).astype(np.uint8)

            elif algo == "Max Filter":
                # فیلتر آماری ماکزیمم (یا فرسایش خاکستری معکوس) مناسب نویزهای تیره
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                kernel = np.ones((k_size, k_size), np.uint8)
                img = cv2.dilate(img, kernel)

            elif algo == "Min Filter":
                # فیلتر آماری مینیمم مناسب برای نویزهای روشن و نمک
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                kernel = np.ones((k_size, k_size), np.uint8)
                img = cv2.erode(img, kernel)

            elif algo == "Midpoint Filter":
                # فیلتر نقطه میانی برای هموارسازی و بازسازی نویزها
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                kernel = np.ones((k_size, k_size), np.uint8)
                mx = cv2.dilate(img, kernel).astype(np.float32)
                mn = cv2.erode(img, kernel).astype(np.float32)
                img = np.clip((mx + mn) / 2.0, 0, 255).astype(np.uint8)

            elif algo == "Alpha-trimmed Mean Filter":
                # فیلتر میانگین آلفا-بریده مناسب برای نویزهای ترکیبی چندگانه
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                d_val = p2
                h, w = gray.shape
                pad = k_size // 2
                padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                out = np.zeros_like(gray, dtype=np.float32)
                for r in range(h):
                    for c in range(w):
                        window = padded[r:r+k_size, c:c+k_size].flatten()
                        window.sort()
                        if d_val > 0 and d_val < len(window):
                            trimmed = window[d_val//2 : -d_val//2]
                        else:
                            trimmed = window
                        out[r, c] = trimmed.mean()
                res = np.clip(out, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            elif algo == "Adaptive Local Noise Reduction":
                # فیلتر وفقی کاهش نویز محلی بر اساس واریانس و میانگین آماری محله
                k_size = p1 if p1 % 2 == 1 else p1 + 1
                noise_var = p2 if p2 > 0 else 100
                img_f = gray.astype(np.float32)
                local_mean = cv2.blur(img_f, (k_size, k_size))
                local_sq_mean = cv2.blur(img_f**2, (k_size, k_size))
                local_var = np.maximum(0, local_sq_mean - local_mean**2)
                ratio = noise_var / (local_var + 1e-8)
                ratio = np.clip(ratio, 0.0, 1.0)
                res = img_f - ratio * (img_f - local_mean)
                img_res = np.clip(res, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB)

            elif algo == "Adaptive Median Filter":
                # فیلتر میانه وفقی پیشرفته با اندازه پنجره متغیر متغیر جهت رفع شدت نویز بالا
                k_max = p1 if p1 % 2 == 1 else p1 + 1
                if k_max < 3: k_max = 7
                h, w = gray.shape
                out = np.zeros_like(gray)
                pad = k_max // 2
                padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                for r in range(h):
                    for c in range(w):
                        k = 3
                        while k <= k_max:
                            offset = pad - k // 2
                            window = padded[r+offset : r+offset+k, c+offset : c+offset+k]
                            z_min = np.min(window)
                            z_max = np.max(window)
                            z_med = np.median(window)
                            z_xy = gray[r, c]
                            a1 = float(z_med) - float(z_min)
                            a2 = float(z_med) - float(z_max)
                            if a1 > 0 and a2 < 0:
                                b1 = float(z_xy) - float(z_min)
                                b2 = float(z_xy) - float(z_max)
                                if b1 > 0 and b2 < 0:
                                    out[r, c] = z_xy
                                else:
                                    out[r, c] = z_med
                                break
                            else:
                                k += 2
                        else:
                            out[r, c] = z_med
                img = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

            elif algo in ["Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"]:
                # عملیات‌های فرسایش، برجسته‌سازی، باز و بستن مورفولوژیکی
                kernel_size = p1 if p1 % 2 == 1 else p1 + 1
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                if algo == "Erosion":
                    img = cv2.erode(img, kernel, iterations=1)
                elif algo == "Dilation":
                    img = cv2.dilate(img, kernel, iterations=1)
                elif algo == "Opening":
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                elif algo == "Closing":
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                elif algo == "Morphological Gradient":
                    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

            elif algo == "Global Thresholding":
                # آستانه‌گذاری سراسری
                _, thresh = cv2.threshold(gray, p1, 255, cv2.THRESH_BINARY)
                img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            elif algo == "Adaptive Mean Threshold":
                # آستانه‌گذاری محلی میانگین
                block_size = p1 if p1 % 2 == 1 else p1 + 1
                if block_size < 3: block_size = 3
                c_val = p2
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_val)
                img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            elif algo == "Adaptive Gaussian Threshold":
                # آستانه‌گذاری محلی وفقی گاوسی
                block_size = p1 if p1 % 2 == 1 else p1 + 1
                if block_size < 3: block_size = 3
                c_val = p2
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val)
                img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            elif algo == "Otsu Thresholding":
                # آستانه‌گذاری خودکار بهینه اتسو
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            elif algo == "RGB to HSV":
                # تبدیل فضای رنگی RGB به HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img = hsv 
            
            elif algo == "RGB to LAB":
                # تبدیل فضای رنگی RGB به LAB
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                img = lab

            elif algo == "RGB to YCrCb":
                # تبدیل فضای رنگی RGB به YCrCb
                ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                img = ycrcb

            # Save result & update UI
            self.processed_image = img
            self.show_images()
            self.update_info(self.processed_image)

        except Exception as e:
            messagebox.showerror("خطا در اجرا", f"الگوریتم با خطا مواجه شد:\n{str(e)}")
            if self.history:
                self.original_image = self.history.pop()

    def show_histogram(self):
        if self.processed_image is None: return
        
        hist_window = cv_gui.Toplevel(self.root)
        hist_window.title("نمودار هیستوگرام تصویر پردازش شده")
        hist_window.geometry("650x450")
        hist_window.transient(self.root)

        # Use clean Figures interface to prevent memory leaks in Tkinter
        fig = Figure(figsize=(6, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        
        if len(self.processed_image.shape) == 3:
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                ax.plot(hist, color=color, label=f"کانال {color.upper()}")
            ax.set_title("Color Channel Histogram")
            ax.legend()
        else:
            hist = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
            ax.plot(hist, color='black', label="شدت روشنایی")
            ax.set_title("Grayscale Histogram")
            ax.legend()
            
        ax.set_xlim([0, 256])
        ax.grid(True, linestyle='--', alpha=0.5)
        
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


if __name__ == "__main__":
    root = cv_gui.Tk()
    app = AdvancedVisionApp(root)
    root.mainloop()
