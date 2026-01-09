import torch
import torch.nn as nn
from torchvision import models, transforms
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import random

# ==========================================
# CẤU HÌNH BLACK-BOX ATTACK (Tấn công hộp đen)
# ==========================================
# Trong tấn công Black-box, kẻ tấn công KHÔNG biết thông tin bên trong mô hình:
# - KHÔNG biết Gradient.
# - KHÔNG biết trọng số (weights).
# - Chỉ có thể gửi đầu vào (Input) và nhận đầu ra (Output probs/class).
#
# Phương pháp sử dụng ở đây: SimBA (Simple Black-box Adversarial Attacks)
# Nguyên lý: Thử thay đổi ngẫu nhiên các pixel, nếu xác suất của nhãn đúng giảm -> giữ lại thay đổi đó.
# ==========================================

device = torch.device("cpu")

# 1. Tải mô hình (Tại đây mô hình đóng vai trò là "Oracle" - Hộp đen trả về kết quả)
print("Đang khởi tạo 'Hộp đen' (MobileNetV2)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Label
url_labels = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(url_labels).json()

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def get_probs(model, x):
    """Hàm giả lập việc gọi API của hộp đen: Input ảnh -> Output xác suất"""
    with torch.no_grad(): # KHÔNG dùng gradient
        output = model(x)
        probs = torch.nn.functional.softmax(output, dim=1)
    return probs

# 3. Phương thức tấn công SimBA (Simple Black-box Adversarial)
def black_box_simba_attack(model, image, target_label_idx, epsilon=0.2, max_iters=500):
    """
    Tấn công SimBA:
    - Chọn ngẫu nhiên 1 hướng (pixel) trong không gian ảnh.
    - Thử cộng/trừ epsilon vào hướng đó.
    - Nếu xác suất nhãn đúng giảm xuống, chấp nhận thay đổi.
    """
    adv_image = image.clone()
    
    # Lấy xác suất ban đầu của nhãn đúng
    probs = get_probs(model, adv_image)
    best_prob = probs[0, target_label_idx].item()
    
    print(f"Bắt đầu tấn công Black-box (SimBA) trong {max_iters} lần thử...")
    print(f"Xác suất ban đầu của nhãn đúng: {best_prob:.4f}")
    
    c, h, w = image.size(1), image.size(2), image.size(3)
    n_dims = c * h * w
    
    # Tạo danh sách các pixel ngẫu nhiên để duyệt qua (duyệt ngẫu nhiên)
    # Để đơn giản, ta chỉ random chọn vector đơn vị chuẩn (thay đổi 1 pixel tại 1 kênh màu)
    
    for i in range(max_iters):
        # 1. Chọn ngẫu nhiên 1 pixel (channel, x, y)
        layer = random.randint(0, c - 1)
        x_pos = random.randint(0, h - 1)
        y_pos = random.randint(0, w - 1)
        
        # Tạo vector nhiễu q (chỉ thay đổi tại 1 điểm)
        # Vì tensor đã normalize, ta cần scale epsilon cho phù hợp hoặc cộng trực tiếp
        # Ở đây cộng trực tiếp vào tensor đã normalize
        
        # Thử hướng dương: + epsilon
        adv_image[0, layer, x_pos, y_pos] += epsilon
        probs_plus = get_probs(model, adv_image)
        prob_plus = probs_plus[0, target_label_idx].item()
        
        if prob_plus < best_prob:
            # Nếu xác suất nhãn đúng giảm -> Tốt, giữ nguyên thay đổi
            best_prob = prob_plus
        else:
            # Nếu không giảm, quay lại và thử hướng âm: - epsilon
            adv_image[0, layer, x_pos, y_pos] -= epsilon # Hoàn tác
            adv_image[0, layer, x_pos, y_pos] -= epsilon # Trừ đi (tổng cộng là trừ epsilon so với gốc)
            
            probs_minus = get_probs(model, adv_image)
            prob_minus = probs_minus[0, target_label_idx].item()
            
            if prob_minus < best_prob:
                best_prob = prob_minus
            else:
                # Nếu cả 2 hướng đều không làm giảm xác suất -> Hoàn tác về như cũ
                adv_image[0, layer, x_pos, y_pos] += epsilon

        # Kiểm tra xem mô hình đã đoán sai chưa
        current_pred = get_probs(model, adv_image).max(1)[1].item()
        if current_pred != target_label_idx:
            print(f"-> Tấn công thành công tại bước {i}!")
            print(f"   Dự đoán mới: {labels[current_pred]} (Prob nhãn gốc: {best_prob:.4f})")
            break
            
        if i % 50 == 0:
            print(f"Iter {i}: Prob nhãn gốc = {best_prob:.4f}")

    return adv_image

# --- MAIN ---

img_url = "https://i.imgur.com/Dvkyrua.jpeg"
response = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'})
orig_image_pil = Image.open(BytesIO(response.content))
input_tensor = preprocess(orig_image_pil).unsqueeze(0).to(device)

# Lấy nhãn đúng ban đầu
probs_orig = get_probs(model, input_tensor)
target_label_idx = probs_orig.max(1)[1].item()
print(f"Dự đoán ban đầu: {labels[target_label_idx]}")

# Thực hiện tấn công
adv_tensor = black_box_simba_attack(model, input_tensor, target_label_idx)

# Kết quả
output_adv = get_probs(model, adv_tensor)
adv_pred_idx = output_adv.max(1)[1].item()
print(f"Dự đoán cuối cùng (Black-box): {labels[adv_pred_idx]}")

# Hiển thị
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(denormalize(input_tensor.squeeze().cpu()).permute(1, 2, 0))
ax[0].set_title(f"GỐC: {labels[target_label_idx]}")
ax[0].axis('off')

ax[1].imshow(denormalize(adv_tensor.squeeze().cpu()).permute(1, 2, 0))
ax[1].set_title(f"BLACK-BOX SIMBA: {labels[adv_pred_idx]}")
ax[1].axis('off')

plt.show()
