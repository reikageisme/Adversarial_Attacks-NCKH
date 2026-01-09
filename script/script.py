import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

device = torch.device("cpu")


model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
model.eval()

url_labels = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(url_labels).json()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def show_images(original, noise, adversarial, pred_orig, pred_adv):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    orig_img = original.squeeze().detach().cpu() * std + mean
    adv_img = adversarial.squeeze().detach().cpu() * std + mean
    noise_img = noise.squeeze().detach().cpu()

    orig_img = torch.clamp(orig_img, 0, 1)
    adv_img = torch.clamp(adv_img, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(orig_img.permute(1, 2, 0))
    ax[0].set_title(f"GỐC: {pred_orig}")
    ax[0].axis('off')

    ax[1].imshow(noise_img.permute(1, 2, 0), cmap='gray')
    ax[1].set_title("NHIỄU (Đã phóng đại)")
    ax[1].axis('off')

    ax[2].imshow(adv_img.permute(1, 2, 0))
    ax[2].set_title(f"BỊ TẤN CÔNG: {pred_adv}")
    ax[2].axis('off')

    plt.show()

def pgd_attack(model, image, labels, epsilon, alpha, num_steps):
    """
    Projected Gradient Descent (PGD) Attack
    Mạnh hơn FGSM bằng cách thực hiện nhiều bước nhỏ và điều chỉnh.
    """
    # Clone ảnh gốc để làm tham chiếu giới hạn
    original_image = image.clone().detach()
    
    # Bắt đầu từ ảnh gốc
    adv_image = image.clone().detach()
    
    for i in range(num_steps):
        adv_image.requires_grad = True
        
        # Forward pass
        output = model(adv_image)
        
        # Tính loss
        loss = F.nll_loss(F.log_softmax(output, dim=1), labels)
        
        # Zero grad
        model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Lấy gradient
        data_grad = adv_image.grad.data
        
        # Cập nhật ảnh: x = x + alpha * sign(grad)
        adv_image = adv_image + alpha * data_grad.sign()
        
        # Projection (Cắt): Đảm bảo nhiễu không lệch quá epsilon so với ảnh gốc
        # Clip phần thặng dư (eta) trong khoảng [-epsilon, epsilon]
        eta = torch.clamp(adv_image - original_image, min=-epsilon, max=epsilon)
        
        # Áp dụng lại vào ảnh gốc
        adv_image = original_image + eta
        
        # Detach để không lưu graph cho bước sau (tiết kiệm bộ nhớ)
        adv_image = adv_image.detach()
        
    return adv_image

img_url = "https://i.imgur.com/Dvkyrua.jpeg"

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

response = requests.get(img_url, headers=headers)

if response.status_code == 200:
    orig_image = Image.open(BytesIO(response.content))
    print("Đã tải ảnh thành công!")
else:
    print(f"Lỗi tải ảnh. Mã lỗi: {response.status_code}")

input_tensor = preprocess(orig_image).unsqueeze(0).to(device)

input_tensor.requires_grad = True

# Dự đoán lần 1 (Ảnh sạch)
output = model(input_tensor)
init_pred_idx = output.max(1, keepdim=True)[1].item()
print(f"Dự đoán ban đầu: {labels[init_pred_idx]}")

# Setup parameters cho PGD
epsilon = 0.1         # Giới hạn nhiễu tối đa
alpha = 0.01          # Bước nhảy mỗi lần lặp (thường nhỏ hơn epsilon)
num_steps = 20        # Số lần lặp lại tấn công

# Label đúng để tính loss (muốn làm sai lệch càng xa label này càng tốt)
target_label = torch.tensor([init_pred_idx]).to(device)

# Thực hiện tấn công PGD
perturbed_data = pgd_attack(model, input_tensor, target_label, epsilon, alpha, num_steps)

# Tính lại noise để hiển thị (chỉ mang tính chất minh họa vì đã qua nhiều bước)
data_grad = perturbed_data - input_tensor 

output_adv = model(perturbed_data)
adv_pred_idx = output_adv.max(1, keepdim=True)[1].item()

print(f"Dự đoán sau khi tấn công: {labels[adv_pred_idx]}")
show_images(input_tensor, data_grad, perturbed_data, labels[init_pred_idx], labels[adv_pred_idx])