import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# ==========================================
# CẤU HÌNH WHITE-BOX ATTACK (Tấn công hộp trắng)
# ==========================================
# Trong tấn công White-box, kẻ tấn công có toàn quyền truy cập vào mô hình:
# - Biết kiến trúc mạng (Architecture)
# - Biết tham số trọng số (Weights)
# - Có thể tính toán Gradient (Đạo hàm) để tìm hướng nhiễu tối ưu.
# ==========================================

device = torch.device("cpu")

# 1. Tải mô hình (Giả lập việc hacker có source code của model)
print("Đang tải mô hình MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
model.eval() # Chế độ đánh giá (không train)

# Label số sang tên (ImageNet)
url_labels = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(url_labels).json()

# 2. Xử lý ảnh đầu vào
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def denormalize(tensor):
    """Hàm hỗ trợ hiển thị ảnh đã normalize"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

# 3. Phương thức tấn công PGD (Projected Gradient Descent) - MỘT KỸ THUẬT WHITE-BOX ĐIỂN HÌNH
def white_box_pgd_attack(model, image, target_label, epsilon=0.1, alpha=0.01, num_steps=50):
    """
    Tấn công PGD: Sử dụng Gradient của mô hình để cập nhật nhiễu.
    Đây là White-box vì ta gọi .backward() để lấy gradient.
    """
    # Tạo bản sao của ảnh để làm điểm bắt đầu
    adv_image = image.clone().detach()
    original_image = image.clone().detach()
    
    print(f"Bắt đầu tấn công PGD (White-box) trong {num_steps} bước...")
    
    for step in range(num_steps):
        adv_image.requires_grad = True
        
        # Forward pass: Đưa ảnh qua mô hình
        output = model(adv_image)
        
        # Tính Loss: Đo độ sai lệch so với nhãn đúng
        loss = F.nll_loss(F.log_softmax(output, dim=1), target_label)
        
        # Zero grad
        model.zero_grad()
        
        # Backward pass: ĐÂY LÀ BƯỚC QUAN TRỌNG CỦA WHITE-BOX
        # Tính toán hướng dốc (gradient) để biết thay đổi pixel nào sẽ làm tăng lỗi nhiều nhất
        loss.backward()
        
        # Lấy dấu của gradient (Sign of Gradient)
        data_grad = adv_image.grad.data
        
        # Cập nhật ảnh nhiễu: Đi ngược lại hướng giảm lỗi (để làm tăng lỗi)
        adv_image = adv_image + alpha * data_grad.sign()
        
        # Projection (Cắt): Đảm bảo nhiễu không quá lộ liễu so với ảnh gốc (trong khoảng epsilon)
        eta = torch.clamp(adv_image - original_image, min=-epsilon, max=epsilon)
        adv_image = original_image + eta
        
        # Detach để chuẩn bị cho bước lặp sau
        adv_image = adv_image.detach()
        
        # (Optional) Kiểm tra xem đã lừa được chưa
        if step % 10 == 0:
            pred = output.max(1, keepdim=True)[1].item()
            print(f"Bước {step}: Dự đoán hiện tại = {labels[pred]} (Loss: {loss.item():.4f})")

    return adv_image

# --- MAIN ---

# Tải ảnh
img_url = "https://i.imgur.com/Dvkyrua.jpeg"
response = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'})
orig_image_pil = Image.open(BytesIO(response.content))
input_tensor = preprocess(orig_image_pil).unsqueeze(0).to(device)

# Dự đoán gốc
output_orig = model(input_tensor)
init_pred_idx = output_orig.max(1, keepdim=True)[1].item()
print(f"Dự đoán ban đầu (Ảnh sạch): {labels[init_pred_idx]}")

# Thực hiện tấn công
target_label = torch.tensor([init_pred_idx]).to(device)
adv_tensor = white_box_pgd_attack(model, input_tensor, target_label)

# Dự đoán sau tấn công
output_adv = model(adv_tensor)
adv_pred_idx = output_adv.max(1, keepdim=True)[1].item()
print(f"Dự đoán sau tấn công (White-box): {labels[adv_pred_idx]}")

# Hiển thị
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(denormalize(input_tensor.squeeze().cpu()).permute(1, 2, 0))
ax[0].set_title(f"GỐC: {labels[init_pred_idx]}")
ax[0].axis('off')

ax[1].imshow(denormalize(adv_tensor.squeeze().cpu()).permute(1, 2, 0))
ax[1].set_title(f"WHITE-BOX PGD: {labels[adv_pred_idx]}")
ax[1].axis('off')

plt.show()
