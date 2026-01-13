"""
=================================================================================
PHYSICAL WORLD TESTER - Test Adversarial Patch với Camera/Webcam
=================================================================================
Công cụ này dùng để kiểm tra patch đã in ra giấy có hoạt động trong thế giới thực không.

Cách sử dụng:
1. In patch từ universal_ghost_patch.py
2. Dán patch lên vật thể
3. Chạy script này với webcam để kiểm tra real-time
=================================================================================
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import time

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Load ImageNet labels
import requests
try:
    LABELS = requests.get(
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
        timeout=5
    ).json()
except:
    LABELS = [f"class_{i}" for i in range(1000)]


class PhysicalWorldTester:
    """
    Test adversarial patch trong thế giới thực với webcam.
    """
    
    def __init__(self, model_names: list = None):
        """
        Khởi tạo tester với các model.
        
        Args:
            model_names: Danh sách model để test
        """
        if model_names is None:
            model_names = ['mobilenet']
        
        self.models = {}
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        print("🔧 Loading models for physical world testing...")
        for name in model_names:
            self.models[name] = self._load_model(name)
            print(f"   ✅ Loaded: {name}")
    
    def _load_model(self, name: str):
        """Load model theo tên"""
        if name == 'mobilenet':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif name == 'yolo':
            # Placeholder - YOLO cần setup riêng
            print("⚠️  YOLO requires separate setup (ultralytics)")
            return None
        else:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        model = model.to(DEVICE)
        model.eval()
        return model
    
    def predict(self, frame: np.ndarray) -> dict:
        """
        Dự đoán từ frame camera.
        
        Args:
            frame: OpenCV frame (BGR)
        
        Returns:
            dict: {model_name: (label, confidence)}
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Preprocess
        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(DEVICE)
        
        results = {}
        with torch.no_grad():
            for name, model in self.models.items():
                if model is None:
                    continue
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = probs.max(1)
                results[name] = (LABELS[pred.item()], conf.item())
        
        return results
    
    def run_webcam_test(self, camera_id: int = 0):
        """
        Chạy test real-time với webcam.
        
        Args:
            camera_id: ID của camera (0 = default)
        """
        print("\n📷 Starting webcam test...")
        print("   Press 'q' to quit")
        print("   Press 's' to save screenshot")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            frame_count += 1
            
            # Predict every 5 frames for performance
            if frame_count % 5 == 0:
                predictions = self.predict(frame)
                
                # Draw predictions on frame
                y_offset = 30
                for name, (label, conf) in predictions.items():
                    text = f"{name}: {label} ({conf:.1%})"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
            
            # FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show
            cv2.imshow("Physical World Test - Press 'q' to quit", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"   📸 Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Webcam test ended.")
    
    def test_single_image(self, image_path: str):
        """
        Test một ảnh tĩnh.
        
        Args:
            image_path: Đường dẫn đến ảnh
        """
        print(f"\n🖼️  Testing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Cannot load image: {image_path}")
            return
        
        # Predict
        predictions = self.predict(frame)
        
        print("\n📊 Predictions:")
        for name, (label, conf) in predictions.items():
            print(f"   {name}: {label} ({conf:.1%})")
        
        # Show with predictions
        y_offset = 30
        for name, (label, conf) in predictions.items():
            text = f"{name}: {label} ({conf:.1%})"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        cv2.imshow("Image Test - Press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class PatchPrintPreparer:
    """
    Chuẩn bị patch để in ra giấy.
    """
    
    @staticmethod
    def prepare_for_print(patch_path: str, 
                          output_path: str = "print_ready_patch.png",
                          print_size_cm: float = 5.0,
                          dpi: int = 300):
        """
        Chuẩn bị patch để in với kích thước và DPI phù hợp.
        
        Args:
            patch_path: Đường dẫn đến file patch
            output_path: Đường dẫn output
            print_size_cm: Kích thước in (cm)
            dpi: Độ phân giải in
        """
        from PIL import Image
        
        # Load patch
        patch = Image.open(patch_path)
        
        # Tính pixel size cần thiết
        # 1 inch = 2.54 cm
        inches = print_size_cm / 2.54
        pixel_size = int(inches * dpi)
        
        # Resize với chất lượng cao
        patch_resized = patch.resize((pixel_size, pixel_size), Image.LANCZOS)
        
        # Tạo ảnh với border (dễ cắt sau khi in)
        border = 50
        final_size = pixel_size + 2 * border
        final_img = Image.new('RGB', (final_size, final_size), 'white')
        final_img.paste(patch_resized, (border, border))
        
        # Vẽ crop marks (đường cắt)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(final_img)
        
        # Góc trên trái
        draw.line([(0, border), (border//2, border)], fill='black', width=1)
        draw.line([(border, 0), (border, border//2)], fill='black', width=1)
        
        # Góc trên phải
        draw.line([(final_size - border//2, border), (final_size, border)], fill='black', width=1)
        draw.line([(final_size - border, 0), (final_size - border, border//2)], fill='black', width=1)
        
        # Góc dưới trái
        draw.line([(0, final_size - border), (border//2, final_size - border)], fill='black', width=1)
        draw.line([(border, final_size - border//2), (border, final_size)], fill='black', width=1)
        
        # Góc dưới phải
        draw.line([(final_size - border//2, final_size - border), (final_size, final_size - border)], fill='black', width=1)
        draw.line([(final_size - border, final_size - border//2), (final_size - border, final_size)], fill='black', width=1)
        
        # Lưu
        final_img.save(output_path, dpi=(dpi, dpi))
        
        print(f"✅ Print-ready patch saved: {output_path}")
        print(f"   📏 Print size: {print_size_cm}x{print_size_cm} cm")
        print(f"   🖼️  Resolution: {pixel_size}x{pixel_size} px @ {dpi} DPI")
        print(f"   ✂️  Crop marks included for easy cutting")
    
    @staticmethod
    def create_test_sheet(patch_path: str,
                          output_path: str = "test_sheet.png",
                          sizes_cm: list = None):
        """
        Tạo sheet với nhiều kích thước patch để test.
        
        Args:
            patch_path: Đường dẫn patch gốc
            output_path: Đường dẫn output
            sizes_cm: Danh sách kích thước (cm)
        """
        from PIL import Image
        
        if sizes_cm is None:
            sizes_cm = [3, 5, 7, 10]
        
        dpi = 300
        
        # Load patch
        patch = Image.open(patch_path)
        
        # Tạo sheet A4 (210 x 297 mm)
        a4_width_px = int(21 / 2.54 * dpi)  # 21cm
        a4_height_px = int(29.7 / 2.54 * dpi)  # 29.7cm
        
        sheet = Image.new('RGB', (a4_width_px, a4_height_px), 'white')
        
        # Đặt các patch
        x_offset = 50
        y_offset = 50
        max_height = 0
        
        for size_cm in sizes_cm:
            pixel_size = int(size_cm / 2.54 * dpi)
            resized = patch.resize((pixel_size, pixel_size), Image.LANCZOS)
            
            # Kiểm tra có đủ chỗ không
            if x_offset + pixel_size > a4_width_px - 50:
                x_offset = 50
                y_offset += max_height + 100
                max_height = 0
            
            # Paste
            sheet.paste(resized, (x_offset, y_offset))
            
            # Add label
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(sheet)
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
            draw.text((x_offset, y_offset + pixel_size + 10), 
                     f"{size_cm}cm", fill='black', font=font)
            
            x_offset += pixel_size + 80
            max_height = max(max_height, pixel_size + 50)
        
        sheet.save(output_path, dpi=(dpi, dpi))
        print(f"✅ Test sheet saved: {output_path}")
        print(f"   📏 Sizes included: {sizes_cm} cm")


# ==========================================
# BATCH TESTING
# ==========================================

def batch_test_images(image_folder: str, model_names: list = None):
    """
    Test hàng loạt ảnh trong một folder.
    
    Args:
        image_folder: Đường dẫn folder chứa ảnh
        model_names: Danh sách model
    """
    import os
    import glob
    
    tester = PhysicalWorldTester(model_names)
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not images:
        print(f"❌ No images found in {image_folder}")
        return
    
    print(f"\n📂 Found {len(images)} images")
    
    results = []
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        predictions = tester.predict(frame)
        results.append({
            'image': os.path.basename(img_path),
            'predictions': predictions
        })
        
        # Print result
        print(f"\n📷 {os.path.basename(img_path)}:")
        for name, (label, conf) in predictions.items():
            print(f"   {name}: {label} ({conf:.1%})")
    
    return results


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║       PHYSICAL WORLD TESTER - Adversarial Patch Testing        ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Các chức năng:                                                ║
    ║    1. run_webcam_test()  - Test real-time với webcam           ║
    ║    2. test_single_image() - Test một ảnh tĩnh                  ║
    ║    3. prepare_for_print() - Chuẩn bị patch để in               ║
    ║    4. create_test_sheet() - Tạo sheet nhiều kích thước         ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Demo
    print("\n🎮 Demo: Testing with webcam...")
    print("   (Bấm 'q' để thoát)\n")
    
    tester = PhysicalWorldTester(['mobilenet'])
    
    # Uncomment để chạy:
    # tester.run_webcam_test()
    
    # Hoặc test ảnh:
    # tester.test_single_image("path/to/your/image.jpg")
    
    # Hoặc chuẩn bị patch để in:
    # PatchPrintPreparer.prepare_for_print("adversarial_patch.png", "print_ready.png", print_size_cm=5.0)
