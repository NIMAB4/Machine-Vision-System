import tkinter as cv_gui
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class AdvancedVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("سامانه جامع بینایی ماشین و پردازش تصویر")
        self.root.geometry("1300x800")
        self.root.state('zoomed')

        self.original_image = None
        self.processed_image = None
        self.display_image_ref = None
        self.history = []
            
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=('B Nazanin', 10, 'bold'), padding=5)
        style.configure("TLabel", font=('B Nazanin', 11))
        style.configure("Header.TLabel", font=('B Nazanin', 12, 'bold'), foreground="#333")

        self.create_layout()

    def create_layout(self):
        top_frame = ttk.Frame(self.root, padding=10, relief="raised")
        top_frame.pack(side="top", fill="x")

        ttk.Button(top_frame, text="بارگذاری تصویر", command=self.load_image).pack(side="right", padx=5)
        ttk.Button(top_frame, text="ذخیره تصویر", command=self.save_image).pack(side="right", padx=5)
        ttk.Button(top_frame, text="بازگشت", command=self.undo_action).pack(side="left", padx=5)
        ttk.Button(top_frame, text="ریست", command=self.reset_image).pack(side="left", padx=5)
        ttk.Button(top_frame, text="نمایش هیستوگرام", command=self.show_histogram).pack(side="left", padx=5)

        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        control_frame = ttk.Frame(main_paned, width=300, relief="groove", padding=10)
        main_paned.add(control_frame, weight=1)

        ttk.Label(control_frame, text="تنظیمات الگوریتم‌ها", style="Header.TLabel").pack(pady=(0, 10))

        ttk.Label(control_frame, text="انتخاب فصل/دسته:").pack(anchor="ne")
        self.category_var = cv_gui.StringVar()
        self.categories = {
            "فصل 3: تبدیل شدت و فضایی": ["Grayscale", "Negative", "Log Transformation", "Gamma Correction", "Histogram Equalization"],
            "فصل 3: فیلترهای مکانی (هموارسازی)": ["Gaussian Blur", "Median Blur", "Average Blur", "Bilateral Filter"],
            "فصل 3: فیلترهای مکانی (تیزکردن/لبه)": ["Laplacian", "Sobel X", "Sobel Y", "Sobel Combined", "Canny Edge Detector"],
            "فصل 4: حوزه فرکانس": ["FFT Spectrum", "Ideal LowPass", "Ideal HighPass"],
            "فصل 5: بازیابی تصویر": ["Add Salt&Pepper Noise", "Add Gaussian Noise"],
            "فصل 9: پردازش مورفولوژی": ["Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"],
            "فصل 10: قطعه‌بندی": ["Global Thresholding", "Adaptive Mean Threshold", "Adaptive Gaussian Threshold", "Otsu Thresholding"],
            "فصل 6: فضای رنگ": ["RGB to HSV", "RGB to LAB", "RGB to YCrCb"],
            "کاربردی: تشخیص اشکال": ["Find Contours", "Hough Line Transform"]
        }
        self.combo_category = ttk.Combobox(control_frame, textvariable=self.category_var, values=list(self.categories.keys()), state="readonly")
        self.combo_category.pack(fill="x", pady=5)
        self.combo_category.bind("<<ComboboxSelected>>", self.update_algorithm_list)

        ttk.Label(control_frame, text="انتخاب الگوریتم:").pack(anchor="ne")
        self.algo_var = cv_gui.StringVar()
        self.combo_algo = ttk.Combobox(control_frame, textvariable=self.algo_var, state="readonly")
        self.combo_algo.pack(fill="x", pady=5)
        self.combo_algo.bind("<<ComboboxSelected>>", self.reset_params)

        self.lbl_param1 = ttk.Label(control_frame, text="پارامتر 1:")
        self.lbl_param1.pack(anchor="ne", pady=(15, 0))
        self.slider_param1 = ttk.Scale(control_frame, from_=1, to=100, orient="horizontal")
        self.slider_param1.pack(fill="x")
        
        self.lbl_param2 = ttk.Label(control_frame, text="پارامتر 2:")
        self.lbl_param2.pack(anchor="ne", pady=(10, 0))
        self.slider_param2 = ttk.Scale(control_frame, from_=1, to=100, orient="horizontal")
        self.slider_param2.pack(fill="x")

        ttk.Button(control_frame, text="اعمال تغییرات", command=self.apply_algorithm).pack(fill="x", pady=20)
        
        self.info_label = ttk.Label(control_frame, text="اطلاعات تصویر: ...", justify="right", wraplength=280)
        self.info_label.pack(side="bottom", pady=10)

        display_frame = ttk.Frame(main_paned, relief="sunken", padding=5)
        main_paned.add(display_frame, weight=4)

        self.canvas_area = cv_gui.Canvas(display_frame, bg="#2b2b2b")
        self.canvas_area.pack(fill="both", expand=True)

    def update_algorithm_list(self, event):
        selected_cat = self.category_var.get()
        self.combo_algo['values'] = self.categories[selected_cat]
        self.combo_algo.current(0)
        self.reset_params(None)

    def reset_params(self, event):
        algo = self.algo_var.get()
        if "Blur" in algo:
            self.slider_param1.config(from_=1, to=50, value=5)
            self.lbl_param1.config(text="اندازه هسته:")
            self.slider_param2.state(['disabled'])
        elif "Canny" in algo:
            self.slider_param1.config(from_=0, to=255, value=100)
            self.lbl_param1.config(text="آستانه پایین:")
            self.slider_param2.state(['!disabled'])
            self.slider_param2.config(from_=0, to=255, value=200)
            self.lbl_param2.config(text="آستانه بالا:")
        elif "Threshold" in algo:
            self.slider_param1.config(from_=0, to=255, value=127)
            self.lbl_param1.config(text="مقدار آستانه:")
            self.slider_param2.state(['disabled'])
        elif "Gamma" in algo:
            self.slider_param1.config(from_=1, to=50, value=10)
            self.lbl_param1.config(text="مقدار گاما:")
            self.slider_param2.state(['disabled'])
        else:
            self.slider_param1.config(from_=1, to=255, value=10)
            self.lbl_param1.config(text="پارامتر شدت:")
            self.slider_param2.state(['disabled'])

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.bmp;*.jpeg;*.tif")])
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("خطا", "تصویر بارگذاری نشد.")
                return
            
            self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.processed_image = self.original_image.copy()
            self.history = []
            self.show_image(self.processed_image)
            self.update_info(self.processed_image)

    def show_image(self, img_array):
        if img_array is None: return
        
        canvas_width = self.canvas_area.winfo_width()
        canvas_height = self.canvas_area.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 800
            canvas_height = 600

        pil_image = Image.fromarray(img_array)
        
        img_width, img_height = pil_image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (int(img_width*ratio), int(img_height*ratio))
        
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        self.display_image_ref = ImageTk.PhotoImage(pil_image)
        
        self.canvas_area.delete("all")
        self.canvas_area.create_image(canvas_width//2, canvas_height//2, image=self.display_image_ref, anchor="center")

    def save_image(self):
        if self.processed_image is None: return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            save_img = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, save_img)
            messagebox.showinfo("موفق", "تصویر ذخیره شد.")

    def undo_action(self):
        if self.history:
            self.processed_image = self.history.pop()
            self.show_image(self.processed_image)
            self.update_info(self.processed_image)
        else:
            messagebox.showwarning("هشدار", "تاریخچه‌ای وجود ندارد.")

    def reset_image(self):
        if self.original_image is not None:
            self.history.append(self.processed_image.copy())
            self.processed_image = self.original_image.copy()
            self.show_image(self.processed_image)

    def update_info(self, img):
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        dtype = img.dtype
        stats = f"ابعاد: {w}x{h}\nکانال‌ها: {c}\nنوع داده: {dtype}\nمیانگین شدت: {np.mean(img):.2f}"
        self.info_label.config(text=stats)

    def apply_algorithm(self):
        if self.processed_image is None:
            messagebox.showwarning("خطا", "لطفا ابتدا تصویری بارگذاری کنید.")
            return

        algo = self.algo_var.get()
        p1 = int(self.slider_param1.get())
        p2 = int(self.slider_param2.get())

        self.history.append(self.processed_image.copy())

        try:
            img = self.processed_image.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

            if algo == "Grayscale":
                if len(img.shape) == 3:
                    res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            elif algo == "Negative":
                img = 255 - img

            elif algo == "Log Transformation":
                c = 255 / np.log(1 + np.max(img))
                log_image = c * (np.log(img + 1))
                img = np.array(log_image, dtype=np.uint8)

            elif algo == "Gamma Correction":
                gamma = p1 / 10.0
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                img = cv2.LUT(img, table)

            elif algo == "Histogram Equalization":
                if len(img.shape) == 3:
                    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
                else:
                    img = cv2.equalizeHist(img)

            elif algo == "Gaussian Blur":
                k = p1 if p1 % 2 == 1 else p1 + 1
                img = cv2.GaussianBlur(img, (k, k), 0)

            elif algo == "Median Blur":
                k = p1 if p1 % 2 == 1 else p1 + 1
                img = cv2.medianBlur(img, k)
            
            elif algo == "Bilateral Filter":
                img = cv2.bilateralFilter(img, 9, p1*2, p1/2)

            elif algo == "Canny Edge Detector":
                edges = cv2.Canny(img, p1, p2)
                img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            elif algo == "Sobel X":
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                abs_sobel64f = np.absolute(sobelx)
                img = np.uint8(abs_sobel64f)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "Laplacian":
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                img = cv2.convertScaleAbs(laplacian)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            elif algo == "FFT Spectrum":
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift))
                magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)

            elif algo == "Ideal LowPass":
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

            elif algo == "Add Salt&Pepper Noise":
                prob = p1 / 1000.0
                output = np.copy(img)
                num_salt = np.ceil(prob * img.size * 0.5)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
                output[tuple(coords)] = 255
                num_pepper = np.ceil(prob * img.size * 0.5)
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
                output[tuple(coords)] = 0
                img = output

            elif algo in ["Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"]:
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
                _, thresh = cv2.threshold(gray, p1, 255, cv2.THRESH_BINARY)
                img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            elif algo == "Otsu Thresholding":
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            elif algo == "RGB to HSV":
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img = hsv 
            
            elif algo == "Find Contours":
                ret, thresh = cv2.threshold(gray, 127, 255, 0)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            
            self.processed_image = img
            self.show_image(self.processed_image)
            self.update_info(self.processed_image)

        except Exception as e:
            messagebox.showerror("خطا در اجرا", f"الگوریتم با خطا مواجه شد:\n{str(e)}")
            if self.history:
                self.processed_image = self.history.pop()

    def show_histogram(self):
        if self.processed_image is None: return
        
        hist_window = cv_gui.Toplevel(self.root)
        hist_window.title("هیستوگرام تصویر")
        hist_window.geometry("600x400")
        
        fig, ax = plt.subplots()
        
        if len(self.processed_image.shape) == 3:
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                ax.plot(hist, color=color)
            ax.set_title("Color Histogram")
        else:
            hist = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
            ax.plot(hist, color='black')
            ax.set_title("Grayscale Histogram")
            
        ax.set_xlim([0, 256])
        
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = cv_gui.Tk()
    app = AdvancedVisionApp(root)
    root.mainloop()
