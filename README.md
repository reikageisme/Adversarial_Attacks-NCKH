# 🔮 Universal Ghost Patch - Adversarial Attack Framework

## Đề tài nghiên cứu
**"Nghiên cứu và triển khai tấn công đối kháng vật lý (Physical Adversarial Attacks) đa mô hình trên thiết bị biên (Edge Devices)."**

*(Research and Implementation of Universal Physical Adversarial Attacks on Edge Devices.)*

---

## 📋 Tổng quan

Framework này tích hợp 4 giả thuyết (Hypotheses) chính:

| Hypothesis | Tên | Mô tả |
|------------|-----|-------|
| **H1** | Black-box Optimization | Sử dụng Genetic Algorithm (GA) / PSO để tối ưu patch mà không cần gradient |
| **H2** | Transferability | Ensemble Attack - đánh lừa nhiều model cùng lúc |
| **H3** | Semantic Constraints | Patch có tính nghệ thuật, nhìn tự nhiên (không phải nhiễu ngẫu nhiên) |
| **H4** | Physical World Attack | EOT (Expectation Over Transformation) - hoạt động khi in ra giấy |

---

## 🗂️ Cấu trúc Project

```
script/
├── generate_patch.py           # 🔥 Tạo adversarial patch (EOT + Ensemble)
├── test_attack_webcam.py       # 🎮 Demo real-time với webcam
├── universal_ghost_patch.py    # Framework đầy đủ (H1-H4 với GA)
├── physical_world_tester.py    # Test in ấn & Physical Attack
├── white_box_attack.py         # Tấn công White-box (PGD) 
├── black_box_attack.py         # Tấn công Black-box (SimBA)
├── script.py                   # Code cơ bản ban đầu
├── data/                       # 📂 Thư mục chứa ảnh training
└── README.md                   # File này
```

---

## ⚡ Bắt đầu nhanh (Quick Start)

### 1. Cài đặt dependencies

```bash
pip install requirements.txt
```

### 2. Tạo thư mục data và thêm ảnh training

```bash
mkdir data
# Bỏ 10-20 ảnh người vào thư mục data/
```

### 3. Tạo Adversarial Patch

```bash
python generate_patch.py
```

Output:
- `adversarial_patch.png` - Patch nhỏ để test
- `adversarial_patch_printable.png` - Patch lớn để IN RA GIẤY
- `training_visualization.png` - Biểu đồ training

### 4. Demo với Webcam

```bash
python test_attack_webcam.py
```

Phím điều khiển:
- `t` - Bật/Tắt Attack mode
- `s` - Chụp screenshot  
- `q` - Thoát

### 5. (Optional) Quick Test các component

```python
from universal_ghost_patch import quick_test
quick_test()
```

---

## 🔬 Chi tiết kỹ thuật

### EOT (Expectation Over Transformation) - Chìa khóa H4

EOT là kỹ thuật quan trọng nhất để patch hoạt động trong thế giới thực:

```python
# Các biến đổi được giả lập trong quá trình training:
EOT_ROTATION_RANGE = (-30, 30)       # Xoay từ -30 đến 30 độ
EOT_SCALE_RANGE = (0.15, 0.4)        # Patch chiếm 15-40% ảnh
EOT_BRIGHTNESS_RANGE = (0.7, 1.3)    # Độ sáng 70%-130%
EOT_NOISE_LEVEL = 0.05               # Mức nhiễu Gaussian
```

### Ensemble Attack (H2 - Transferability)

Để patch đánh lừa được nhiều model:

```python
# Trong generate_patch.py
USE_ENSEMBLE = True
ENSEMBLE_MODELS = ['mobilenet', 'resnet50', 'inception', 'vgg16']
```

### Target Classes (ImageNet)

Một số class thú vị để thử:
| Class ID | Tên | Mô tả |
|----------|-----|-------|
| 859 | toaster | Lò nướng (mặc định) |
| 954 | banana | Quả chuối |
| 508 | computer keyboard | Bàn phím |
| 703 | park bench | Ghế công viên |
| 281 | tabby cat | Mèo mướp |

---

## 📊 Workflow hoàn chỉnh
PatchPrintPreparer.create_test_sheet(
    "adversarial_patch.png",
    "test_sheet.png",
    sizes_cm=[3, 5, 7, 10]
)
```

---

## 🧬 Chi tiết kỹ thuật

### H1: Genetic Algorithm Optimizer

```python
@dataclass
class Circle:
    """Gen = [x, y, radius, R, G, B, alpha] của mỗi hình tròn"""
    x: float        # Vị trí x (0-1)
    y: float        # Vị trí y (0-1)
    radius: float   # Bán kính (0-0.5)
    r, g, b: int    # Màu RGB (0-255)
    alpha: int      # Độ trong suốt (0-255)
```

**Quy trình tiến hóa:**
1. Khởi tạo quần thể N patches ngẫu nhiên
2. Tính fitness = số model bị đánh lừa / tổng số model
3. Selection (Tournament)
4. Crossover (lai ghép circles/rectangles)
5. Mutation (đột biến các thuộc tính)
6. Lặp lại

### H2: Ensemble Models

```python
ensemble = EnsembleModels(['mobilenet', 'resnet50', 'inception', 'vgg16'])

# Fitness function
fitness = ensemble.compute_ensemble_fitness(patched_image, original_labels)
# fitness = 1.0 nếu TẤT CẢ model bị lừa
```

### H3: Semantic Patch

Thay vì tối ưu pixel-by-pixel (nhiễu), patch được tạo từ:
- **Circles**: Hình tròn bán trong suốt
- **Rectangles**: Hình chữ nhật có góc xoay

→ Kết quả: Patch trông như "art abstract" thay vì nhiễu hạt

### H4: EOT (Expectation Over Transformation)

```python
eot = EOTTransformer(
    rotation_range=(-20, 20),      # Xoay ngẫu nhiên
    scale_range=(0.9, 1.1),        # Scale ngẫu nhiên
    brightness_range=(0.8, 1.2),   # Độ sáng thay đổi
    noise_level=0.03,              # Nhiễu Gaussian
    blur_prob=0.2                  # Xác suất bị blur
)
```

Mỗi iteration training, patch được áp dụng với các biến đổi random này → Patch robust hơn khi in ra giấy.

---

## 📊 Workflow hoàn chỉnh

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Initialize Population (N patches)                               │
│              ↓                                                      │
│  2. For each patch:                                                 │
│     a. Apply EOT transforms (rotation, scale, brightness...)        │
│     b. Apply patch to base image at random position                 │
│     c. Query ALL models in ensemble                                 │
│     d. Compute fitness = (models fooled) / (total models)           │
│              ↓                                                      │
│  3. Evolution:                                                      │
│     - Elite selection (keep top 10%)                                │
│     - Tournament selection                                          │
│     - Crossover (mix circles/rectangles)                            │
│     - Mutation (change colors, positions, sizes)                    │
│              ↓                                                      │
│  4. Repeat for G generations                                        │
│              ↓                                                      │
│  5. Output: Best patch with highest fitness                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      PHYSICAL WORLD PHASE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Export patch as high-res PNG (300 DPI)                          │
│              ↓                                                      │
│  2. Print on paper/sticker                                          │
│              ↓                                                      │
│  3. Stick on object                                                 │
│              ↓                                                      │
│  4. Test with webcam/phone camera                                   │
│              ↓                                                      │
│  5. Verify: Model misclassifies object?                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Lưu ý & Thách thức

### Thời gian training
- **White-box (PGD)**: ~5-10 phút với GPU, ~30-60 phút với CPU
- **Black-box (GA)**: ~1-4 giờ tùy population size và số generation

### GPU Memory
- 4 models ensemble cần ~8GB VRAM
- Khuyến nghị: RTX 3060+ hoặc giảm số model

### Tips tối ưu
1. **Bắt đầu với White-box** để có baseline
2. **2 models ensemble** là đủ cho demo
3. **EOT samples = 3-5** là cân bằng tốt giữa chất lượng và tốc độ
4. **Patch size 60-100px** hiệu quả nhất

---

## 📚 Tài liệu tham khảo

1. **Adversarial Patch** - Brown et al., 2017
2. **EOT (Expectation Over Transformation)** - Athalye et al., 2018
3. **Universal Perturbations** - Moosavi-Dezfooli et al., 2017
4. **Black-box Adversarial Attacks** - Chen et al., 2017

---

## 👨‍💻 Author

**ReiKage** - NCKH Project 2025-2026

---

## 📝 License

Educational purposes only. Use responsibly.
