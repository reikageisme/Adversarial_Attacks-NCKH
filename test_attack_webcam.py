"""
=================================================================================
TEST ADVERSARIAL ATTACK - Demo với Webcam
=================================================================================
Code này mở webcam và demo việc tấn công đối kháng real-time.

Tính năng:
- Mở Webcam máy tính
- Load model MobileNetV2/ResNet để nhận diện
- Bật/Tắt chế độ tấn công bằng phím bấm
- Dán miếng adversarial_patch.png vào video webcam
- Hiển thị xác suất theo thời gian thực

Phím điều khiển:
- 't' : Bật/Tắt Attack mode
- 'p' : Thay đổi vị trí patch (center/follow mouse)
- 's' : Chụp screenshot
- 'r' : Reset về trạng thái ban đầu
- 'q' : Thoát

=================================================================================
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import requests
import os
import time

# ==========================================
# CẤU HÌNH
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_PATH = "adversarial_patch.png"
TARGET_CLASS_IDX = 859  # Toaster (Lò nướng)
PATCH_SCALE = 0.25      # Patch chiếm 25% chiều rộng màn hình

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print("=" * 60)
print("🎮 ADVERSARIAL ATTACK DEMO - Webcam Version")
print("=" * 60)
print(f"🖥️  Device: {DEVICE}")

# ==========================================
# 1. LOAD MODEL & LABELS
# ==========================================
print("\n📦 Loading model...")

# Dùng MobileNetV2 cho nhẹ, chạy mượt
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(DEVICE)
model.eval()
print("   ✅ Loaded MobileNetV2")

# Optional: Load thêm ResNet để so sánh
try:
    model_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
    model_resnet.eval()
    use_resnet = True
    print("   ✅ Loaded ResNet50")
except:
    use_resnet = False

# Tải labels
print("\n📋 Loading labels...")
try:
    url_labels = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(url_labels, timeout=5).json()
    print("   ✅ Loaded ImageNet labels")
except:
    labels = [f"class_{i}" for i in range(1000)]
    print("   ⚠️ Using default labels")

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ==========================================
# 2. LOAD PATCH
# ==========================================
print(f"\n🎨 Loading patch from '{PATCH_PATH}'...")

if os.path.exists(PATCH_PATH):
    patch_img = Image.open(PATCH_PATH).convert('RGB')
    print("   ✅ Loaded adversarial patch")
else:
    print(f"   ⚠️ Không tìm thấy '{PATCH_PATH}'!")
    print("   📝 Hãy chạy 'generate_patch.py' trước để tạo patch")
    print("   🔧 Đang tạo patch placeholder...")
    
    # Tạo patch placeholder (rainbow gradient)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            arr[i, j] = [
                int(127 + 127 * np.sin(i / 10)),
                int(127 + 127 * np.sin(j / 10)),
                int(127 + 127 * np.sin((i + j) / 15))
            ]
    patch_img = Image.fromarray(arr)

patch_np_original = np.array(patch_img)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def predict(frame, model, with_patch=False, patch_np=None, patch_pos=None):
    """
    Chạy inference trên frame.
    
    Args:
        frame: OpenCV frame (BGR)
        model: PyTorch model
        with_patch: Có dán patch không
        patch_np: Numpy array của patch
        patch_pos: Tuple (x, y) vị trí dán
    
    Returns:
        Tuple (pred_idx, confidence, all_probs, processed_frame)
    """
    display_frame = frame.copy()
    process_frame = frame.copy()
    
    if with_patch and patch_np is not None and patch_pos is not None:
        h, w = frame.shape[:2]
        patch_h, patch_w = patch_np.shape[:2]
        x, y = patch_pos
        
        # Đảm bảo không vượt quá biên
        x = max(0, min(x, w - patch_w))
        y = max(0, min(y, h - patch_h))
        
        # Dán patch lên frame để xử lý
        process_frame[y:y+patch_h, x:x+patch_w] = patch_np
        
        # Vẽ khung đỏ trên display frame
        cv2.rectangle(display_frame, (x, y), (x+patch_w, y+patch_h), (0, 0, 255), 3)
        display_frame[y:y+patch_h, x:x+patch_w] = patch_np
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Preprocess và inference
    input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
    
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    
    return pred_idx, confidence, probs.cpu().numpy(), display_frame


def draw_info(frame, attack_mode, pred_idx, confidence, probs, fps):
    """Vẽ thông tin lên frame"""
    h, w = frame.shape[:2]
    
    # Background panel
    cv2.rectangle(frame, (5, 5), (350, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (350, 180), (100, 100, 100), 2)
    
    # Status
    status = "🔴 ATTACK: ON" if attack_mode else "🟢 ATTACK: OFF"
    color = (0, 0, 255) if attack_mode else (0, 255, 0)
    cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Prediction
    pred_label = labels[pred_idx][:20]
    cv2.putText(frame, f"AI Sees: {pred_label}", (15, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (15, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Target class probability
    target_prob = probs[TARGET_CLASS_IDX]
    target_label = labels[TARGET_CLASS_IDX][:15]
    cv2.putText(frame, f"Target ({target_label}): {target_prob:.1%}", (15, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    
    # Progress bar cho target probability
    bar_width = int(300 * target_prob)
    cv2.rectangle(frame, (15, 145), (15 + bar_width, 165), (100, 200, 255), -1)
    cv2.rectangle(frame, (15, 145), (315, 165), (100, 100, 100), 2)
    
    # Instructions
    cv2.putText(frame, "Press 't' to toggle attack | 'q' to quit", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame


def draw_comparison(frame, results_clean, results_attack, attack_mode):
    """Vẽ so sánh kết quả Clean vs Attack"""
    h, w = frame.shape[:2]
    
    if not attack_mode:
        return frame
    
    # Panel bên phải
    panel_x = w - 300
    cv2.rectangle(frame, (panel_x, 5), (w - 5, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, 5), (w - 5, 120), (100, 100, 100), 2)
    
    cv2.putText(frame, "COMPARISON", (panel_x + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Clean result
    clean_label = labels[results_clean[0]][:12]
    cv2.putText(frame, f"Clean: {clean_label} ({results_clean[1]:.0%})", 
                (panel_x + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Attack result
    attack_label = labels[results_attack[0]][:12]
    cv2.putText(frame, f"Attack: {attack_label} ({results_attack[1]:.0%})", 
                (panel_x + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Success indicator
    if results_clean[0] != results_attack[0]:
        cv2.putText(frame, "SUCCESS!", (panel_x + 10, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame


# ==========================================
# 4. MAIN LOOP
# ==========================================
print("\n🎥 Opening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    print("   Thử với camera ID khác (1, 2...) hoặc kiểm tra kết nối")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("   ✅ Webcam opened successfully")
print("\n" + "=" * 60)
print("🎮 CONTROLS:")
print("   't' - Toggle Attack ON/OFF")
print("   's' - Save screenshot")
print("   'r' - Reset")
print("   'q' - Quit")
print("=" * 60)
print("\n🚀 Starting demo... (Press 'q' to quit)\n")

# State variables
attack_mode = False
frame_count = 0
start_time = time.time()
fps = 0

# Resize patch dựa trên kích thước webcam
ret, test_frame = cap.read()
if ret:
    frame_h, frame_w = test_frame.shape[:2]
    patch_size = int(frame_w * PATCH_SCALE)
    patch_np = cv2.resize(patch_np_original, (patch_size, patch_size))
    patch_pos = ((frame_w - patch_size) // 2, (frame_h - patch_size) // 2)
    print(f"   Patch size: {patch_size}x{patch_size} pixels")
    print(f"   Patch position: center ({patch_pos})")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame!")
        break
    
    # Mirror effect (optional - comment out if not needed)
    frame = cv2.flip(frame, 1)
    
    frame_count += 1
    
    # Calculate FPS
    elapsed = time.time() - start_time
    if elapsed > 0:
        fps = frame_count / elapsed
    
    # Get predictions
    # Clean prediction (không có patch)
    pred_clean, conf_clean, probs_clean, _ = predict(frame, model)
    
    # Attack prediction (có patch nếu attack_mode = True)
    pred_attack, conf_attack, probs_attack, display_frame = predict(
        frame, model, 
        with_patch=attack_mode, 
        patch_np=patch_np, 
        patch_pos=patch_pos
    )
    
    # Chọn kết quả để hiển thị
    if attack_mode:
        pred_idx, confidence, probs = pred_attack, conf_attack, probs_attack
    else:
        pred_idx, confidence, probs = pred_clean, conf_clean, probs_clean
        display_frame = frame.copy()
    
    # Draw info
    display_frame = draw_info(display_frame, attack_mode, pred_idx, confidence, probs, fps)
    
    # Draw comparison (khi attack mode ON)
    display_frame = draw_comparison(
        display_frame, 
        (pred_clean, conf_clean), 
        (pred_attack, conf_attack), 
        attack_mode
    )
    
    # Show
    cv2.imshow('Adversarial Attack Demo - Press q to quit', display_frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Exiting...")
        break
    
    elif key == ord('t'):
        attack_mode = not attack_mode
        status = "ON 🔴" if attack_mode else "OFF 🟢"
        print(f"   Attack mode: {status}")
    
    elif key == ord('s'):
        filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, display_frame)
        print(f"   📸 Saved: {filename}")
    
    elif key == ord('r'):
        attack_mode = False
        frame_count = 0
        start_time = time.time()
        print("   🔄 Reset!")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("✅ Demo ended successfully!")
print("=" * 60)
