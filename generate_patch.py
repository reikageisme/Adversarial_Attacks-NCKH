"""
=================================================================================
GENERATE ADVERSARIAL PATCH - Tạo miếng dán đối kháng
=================================================================================
Code này tạo ra file adversarial_patch.png để sử dụng trong Physical Attack.

Kỹ thuật cốt lõi: EOT (Expectation Over Transformation)
- Xoay (Rotation): Giả lập việc dán nghiêng
- Co giãn (Scale): Giả lập người đứng xa/gần  
- Nhiễu (Noise): Giả lập chất lượng camera kém/ánh sáng môi trường
- Vị trí ngẫu nhiên: Giả lập patch ở các vị trí khác nhau

Target: Biến "Person" thành "Toaster" (Class 859) hoặc class khác

Cách chạy:
1. Tạo thư mục 'data/' và bỏ vào 10-20 ảnh người làm background
2. Chạy: python generate_patch.py
3. Output: adversarial_patch.png
=================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (100, 100)     # Kích thước miếng dán (pixel)
IMG_SIZE = (224, 224)       # Kích thước đầu vào của Model (ResNet/MobileNet dùng 224)
EPOCHS = 500                # Số vòng lặp huấn luyện
LEARNING_RATE = 0.05
TARGET_CLASS = 859          # 859 = "toaster" (lò nướng). Mục tiêu: Biến người -> Lò nướng
                            # Một số class thú vị khác:
                            # 954 = banana, 508 = computer keyboard, 703 = park bench
                            # 281 = tabby cat, 207 = golden retriever

# EOT Configuration
EOT_ROTATION_RANGE = (-30, 30)       # Xoay từ -30 đến 30 độ
EOT_SCALE_RANGE = (0.15, 0.4)        # Patch chiếm 15-40% ảnh
EOT_BRIGHTNESS_RANGE = (0.7, 1.3)    # Độ sáng 70%-130%
EOT_NOISE_LEVEL = 0.05               # Mức nhiễu Gaussian
EOT_SAMPLES_PER_ITER = 5             # Số biến thể EOT mỗi iteration

# Ensemble Models để tăng Transferability (H2)
USE_ENSEMBLE = True
ENSEMBLE_MODELS = ['mobilenet', 'resnet50']  # Có thể thêm 'inception', 'vgg16'

print("=" * 60)
print("🎯 ADVERSARIAL PATCH GENERATOR")
print("=" * 60)
print(f"🖥️  Device: {DEVICE}")
print(f"📐 Patch size: {PATCH_SIZE}")
print(f"🎯 Target class: {TARGET_CLASS}")
print(f"🔄 Epochs: {EPOCHS}")
print(f"🧬 Ensemble: {ENSEMBLE_MODELS if USE_ENSEMBLE else 'Disabled'}")
print("=" * 60)

# ==========================================
# 2. LOAD MODELS (Ensemble cho Transferability - H2)
# ==========================================
print("\n📦 Loading models...")

models_dict = {}

def load_model(name):
    """Load model theo tên"""
    if name == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif name == 'inception':
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
    elif name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif name == 'densenet':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    model = model.to(DEVICE)
    model.eval()
    return model

if USE_ENSEMBLE:
    for name in ENSEMBLE_MODELS:
        print(f"   Loading {name}...", end=" ")
        models_dict[name] = load_model(name)
        print("✅")
else:
    models_dict['mobilenet'] = load_model('mobilenet')
    print("   Loaded MobileNetV2 ✅")

# ==========================================
# 3. LOAD DATA (Background Images)
# ==========================================
print("\n📂 Loading background images...")

# Chuẩn hóa dữ liệu theo chuẩn ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def load_background_images(data_dir='data'):
    """Load tất cả ảnh từ thư mục data/"""
    images = []
    
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(data_dir, f)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except:
                    pass
    
    if len(images) == 0:
        print("   ⚠️  Không tìm thấy ảnh trong thư mục 'data/'")
        print("   📥 Đang tải ảnh mẫu từ internet...")
        
        # Tải một số ảnh mẫu từ internet
        sample_urls = [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Person 1
            "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400",  # Person 2
            "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400",  # Person 3
        ]
        
        for i, url in enumerate(sample_urls):
            try:
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    images.append(img)
                    print(f"      ✅ Loaded sample image {i+1}")
            except Exception as e:
                print(f"      ❌ Failed to load sample {i+1}: {e}")
        
        # Nếu vẫn không có ảnh, tạo ảnh giả
        if len(images) == 0:
            print("   ⚠️  Tạo ảnh synthetic để demo...")
            for i in range(5):
                # Tạo ảnh gradient ngẫu nhiên
                arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                images.append(Image.fromarray(arr))
    
    print(f"   ✅ Loaded {len(images)} background images")
    return images

background_images = load_background_images()

# ==========================================
# 4. EOT FUNCTIONS (Expectation Over Transformation - H4)
# ==========================================

def apply_eot_transform(patch_tensor, bg_tensor):
    """
    Áp dụng EOT: Dán patch vào ảnh nền với các biến đổi ngẫu nhiên
    để giả lập thế giới thực.
    
    Args:
        patch_tensor: Tensor của patch (3, H, W)
        bg_tensor: Tensor của ảnh nền (3, H, W)
    
    Returns:
        Tensor ảnh đã được dán patch với biến đổi EOT
    """
    # Clone để không ảnh hưởng tensor gốc
    adv_img = bg_tensor.clone()
    patch_aug = patch_tensor.clone()
    
    # 1. Random Rotation (Xoay miếng dán)
    angle = random.uniform(*EOT_ROTATION_RANGE)
    patch_aug = transforms.functional.rotate(
        patch_aug.unsqueeze(0), angle, 
        interpolation=transforms.InterpolationMode.BILINEAR
    ).squeeze(0)
    
    # 2. Random Scale (Co giãn)
    scale_factor = random.uniform(*EOT_SCALE_RANGE)
    new_h = int(IMG_SIZE[0] * scale_factor)
    new_w = int(IMG_SIZE[1] * scale_factor)
    patch_aug = F.interpolate(
        patch_aug.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    # 3. Random Brightness (Độ sáng)
    brightness = random.uniform(*EOT_BRIGHTNESS_RANGE)
    patch_aug = patch_aug * brightness
    
    # 4. Random Noise (Nhiễu)
    if EOT_NOISE_LEVEL > 0:
        noise = torch.randn_like(patch_aug) * EOT_NOISE_LEVEL
        patch_aug = patch_aug + noise
    
    # Clamp về khoảng hợp lệ
    patch_aug = torch.clamp(patch_aug, -3, 3)
    
    # 5. Random Position (Vị trí dán ngẫu nhiên)
    max_x = IMG_SIZE[1] - new_w
    max_y = IMG_SIZE[0] - new_h
    if max_x > 0 and max_y > 0:
        x_pos = random.randint(0, max_x)
        y_pos = random.randint(0, max_y)
    else:
        x_pos, y_pos = 0, 0
    
    # 6. Dán patch lên ảnh nền
    adv_img[:, y_pos:y_pos+new_h, x_pos:x_pos+new_w] = patch_aug
    
    return adv_img


def compute_tv_loss(patch):
    """
    Total Variation Loss - Làm patch mượt hơn (giảm nhiễu hạt)
    Đây là phần của H3 (Semantic) - patch nhìn tự nhiên hơn
    """
    tv_h = torch.sum(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]))
    tv_w = torch.sum(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]))
    return tv_h + tv_w


def compute_ensemble_loss(adv_img, target_class, models_dict):
    """
    Tính loss cho ensemble attack (H2 - Transferability)
    Loss = Tổng loss của tất cả các model
    """
    total_loss = 0
    target_tensor = torch.tensor([target_class], device=DEVICE)
    
    for name, model in models_dict.items():
        # Resize nếu cần (Inception cần 299x299)
        if name == 'inception':
            img = F.interpolate(adv_img.unsqueeze(0), size=(299, 299), mode='bilinear')
        else:
            img = adv_img.unsqueeze(0)
        
        output = model(img)
        
        # Targeted Attack: Tối đa hóa xác suất của target class
        # Loss = -log(prob(target)) = CrossEntropy với target
        loss = -F.cross_entropy(output, target_tensor)
        total_loss += loss
    
    return total_loss / len(models_dict)


# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\n🚀 Bắt đầu tạo Adversarial Patch...")
print(f"   Target: Class {TARGET_CLASS}")

# Khởi tạo patch ngẫu nhiên
# Option 1: Random noise
# patch = torch.rand((3, PATCH_SIZE[0], PATCH_SIZE[1]), device=DEVICE, requires_grad=True)

# Option 2: Gradient-friendly initialization (thường converge nhanh hơn)
patch = torch.zeros((3, PATCH_SIZE[0], PATCH_SIZE[1]), device=DEVICE)
patch = patch + 0.5  # Start from gray
patch = patch + torch.randn_like(patch) * 0.1  # Add small noise
patch = patch.requires_grad_(True)

# Optimizer
optimizer = optim.Adam([patch], lr=LEARNING_RATE)

# Scheduler để giảm learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Training history
loss_history = []
best_loss = float('inf')
best_patch = None

# Labels để hiển thị
try:
    url_labels = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(url_labels, timeout=5).json()
except:
    labels = [f"class_{i}" for i in range(1000)]

print(f"   🎯 Target label: {labels[TARGET_CLASS]}")
print("\n" + "-" * 50)

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    total_epoch_loss = 0
    
    # Chọn ngẫu nhiên một ảnh background (Batch Training)
    bg_image = random.choice(background_images)
    bg_tensor = preprocess(bg_image).to(DEVICE)
    
    # EOT: Chạy nhiều biến thể và lấy trung bình loss
    for _ in range(EOT_SAMPLES_PER_ITER):
        # Áp dụng EOT transform
        adv_image = apply_eot_transform(patch, bg_tensor)
        
        # Tính loss từ ensemble
        attack_loss = compute_ensemble_loss(adv_image, TARGET_CLASS, models_dict)
        total_epoch_loss += attack_loss
    
    # Trung bình loss qua các EOT samples
    avg_attack_loss = total_epoch_loss / EOT_SAMPLES_PER_ITER
    
    # Total Variation Loss (H3 - Semantic smoothness)
    tv_loss = compute_tv_loss(patch)
    
    # Tổng loss = Attack loss + TV regularization
    # TV weight nhỏ để không ảnh hưởng quá nhiều đến hiệu quả tấn công
    total_loss = avg_attack_loss + 0.001 * tv_loss
    
    # Backward & Update
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Clamp patch về khoảng hợp lệ [0, 1] trước khi normalize
    # Trong không gian normalized của ImageNet, giá trị có thể từ -2 đến 3
    with torch.no_grad():
        # Clamp trong không gian pixel (sau denormalize sẽ nằm trong [0,1])
        patch.data.clamp_(-2.5, 2.5)
    
    # Lưu best patch
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        best_patch = patch.clone().detach()
    
    loss_history.append(total_loss.item())
    
    # Progress report
    if epoch % 25 == 0 or epoch == EPOCHS - 1:
        # Kiểm tra prediction hiện tại
        with torch.no_grad():
            test_adv = apply_eot_transform(patch, bg_tensor)
            test_output = list(models_dict.values())[0](test_adv.unsqueeze(0))
            probs = F.softmax(test_output, dim=1)[0]
            pred_idx = probs.argmax().item()
            target_prob = probs[TARGET_CLASS].item()
        
        print(f"Epoch {epoch:4d}/{EPOCHS} | Loss: {total_loss.item():.4f} | "
              f"Target prob: {target_prob:.2%} | Pred: {labels[pred_idx][:15]}")

print("-" * 50)
print("✅ Training hoàn tất!")

# ==========================================
# 6. LƯU KẾT QUẢ
# ==========================================
print("\n💾 Đang lưu kết quả...")

def save_patch(patch_tensor, filename):
    """Lưu patch tensor thành ảnh PNG"""
    # Denormalize
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(patch_tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(patch_tensor.device)
    patch_img = patch_tensor * std + mean
    patch_img = torch.clamp(patch_img, 0, 1)
    
    # Convert to PIL
    patch_np = (patch_img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(patch_np).save(filename)
    return patch_np

# Lưu patch tốt nhất
if best_patch is not None:
    patch_np = save_patch(best_patch, "adversarial_patch.png")
    print("   ✅ Saved: adversarial_patch.png")
else:
    patch_np = save_patch(patch, "adversarial_patch.png")
    print("   ✅ Saved: adversarial_patch.png")

# Lưu patch với kích thước lớn hơn để in
large_patch = F.interpolate(
    (best_patch if best_patch is not None else patch).unsqueeze(0),
    size=(500, 500), mode='bilinear', align_corners=False
).squeeze(0)
save_patch(large_patch, "adversarial_patch_printable.png")
print("   ✅ Saved: adversarial_patch_printable.png (500x500 for printing)")

# ==========================================
# 7. VISUALIZATION
# ==========================================
print("\n📊 Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Loss history
axes[0, 0].plot(loss_history)
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True)

# 2. Final patch
axes[0, 1].imshow(patch_np)
axes[0, 1].set_title(f"Adversarial Patch\nTarget: {labels[TARGET_CLASS]}")
axes[0, 1].axis('off')

# 3. Original image
bg_sample = random.choice(background_images)
bg_np = np.array(bg_sample.resize(IMG_SIZE))
axes[0, 2].imshow(bg_np)
axes[0, 2].set_title("Original Image")
axes[0, 2].axis('off')

# 4-6. Patched images with different EOT
bg_tensor = preprocess(bg_sample).to(DEVICE)
final_patch = best_patch if best_patch is not None else patch

for i, ax in enumerate(axes[1]):
    with torch.no_grad():
        adv_img = apply_eot_transform(final_patch, bg_tensor)
        
        # Denormalize for display
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(DEVICE)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(DEVICE)
        adv_display = adv_img * std + mean
        adv_display = torch.clamp(adv_display, 0, 1)
        
        # Get prediction
        output = list(models_dict.values())[0](adv_img.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_prob = probs[pred_idx].item()
        target_prob = probs[TARGET_CLASS].item()
    
    ax.imshow(adv_display.cpu().permute(1, 2, 0).numpy())
    ax.set_title(f"EOT Sample {i+1}\nPred: {labels[pred_idx][:12]} ({pred_prob:.1%})\n"
                 f"Target: {target_prob:.1%}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("training_visualization.png", dpi=150)
print("   ✅ Saved: training_visualization.png")

plt.show()

# ==========================================
# 8. FINAL EVALUATION
# ==========================================
print("\n" + "=" * 60)
print("📈 FINAL EVALUATION")
print("=" * 60)

# Test trên nhiều ảnh
success_count = 0
total_tests = min(len(background_images), 10)

print(f"\nTesting on {total_tests} images...")

with torch.no_grad():
    for i in range(total_tests):
        bg = background_images[i]
        bg_tensor = preprocess(bg).to(DEVICE)
        
        # Original prediction
        orig_output = list(models_dict.values())[0](bg_tensor.unsqueeze(0))
        orig_pred = orig_output.argmax().item()
        
        # Patched prediction
        adv_img = apply_eot_transform(final_patch, bg_tensor)
        adv_output = list(models_dict.values())[0](adv_img.unsqueeze(0))
        adv_pred = adv_output.argmax().item()
        adv_probs = F.softmax(adv_output, dim=1)[0]
        target_prob = adv_probs[TARGET_CLASS].item()
        
        # Check success
        is_success = adv_pred == TARGET_CLASS or adv_pred != orig_pred
        if is_success:
            success_count += 1
        
        status = "✅" if is_success else "❌"
        print(f"   Image {i+1}: {labels[orig_pred][:12]:12s} -> {labels[adv_pred][:12]:12s} "
              f"(Target: {target_prob:.1%}) {status}")

print(f"\n🎯 Attack Success Rate: {success_count}/{total_tests} ({success_count/total_tests:.1%})")
print("=" * 60)

print("""
📋 HƯỚNG DẪN SỬ DỤNG:

1. File 'adversarial_patch.png' - Patch nhỏ để test digital
2. File 'adversarial_patch_printable.png' - Patch lớn để IN RA GIẤY
3. Chạy 'test_attack_webcam.py' để demo với webcam

💡 Để cải thiện kết quả:
   - Tăng EPOCHS lên 1000-2000
   - Thêm nhiều ảnh vào thư mục 'data/'
   - Thêm models vào ENSEMBLE_MODELS (như 'inception', 'vgg16')
   - Điều chỉnh TARGET_CLASS theo mục tiêu của bạn
""")
