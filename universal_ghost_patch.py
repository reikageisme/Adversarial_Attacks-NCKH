"""
=================================================================================
UNIVERSAL GHOST PATCH - Adversarial Patch Attack Framework
=================================================================================
Đề tài: "Nghiên cứu và xây dựng phương thức tấn công đối kháng vật lý đa mô hình 
        sử dụng giải thuật tối ưu hóa hộp đen"
        (Research and Development of Universal Physical Adversarial Attacks 
        using Black-box Optimization)

Tích hợp 4 Hypotheses:
- H1: Black-box Optimization (Genetic Algorithm / PSO)
- H2: Transferability (Ensemble Attack - Multi-model)
- H3: Semantic Constraints (Artistic/Natural-looking patches)
- H4: Physical World Attack (EOT - Expectation Over Transformation)

Author: ReiKage
=================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import random
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PHẦN 1: CẤU HÌNH VÀ THIẾT LẬP
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Đang sử dụng: {DEVICE}")

# Tải ImageNet labels
URL_LABELS = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
try:
    LABELS = requests.get(URL_LABELS, timeout=10).json()
except:
    LABELS = [f"class_{i}" for i in range(1000)]

# Transform chuẩn cho ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def denormalize(tensor):
    """Chuyển tensor đã normalize về ảnh gốc để hiển thị"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)


# ==========================================
# PHẦN 2: H2 - ENSEMBLE MODEL (Multi-model Attack)
# ==========================================

class EnsembleModels:
    """
    Tập hợp nhiều model để thực hiện Ensemble Attack.
    Miếng dán phải đánh lừa được TẤT CẢ các model cùng lúc.
    """
    
    def __init__(self, model_names: List[str] = None):
        """
        Khởi tạo ensemble với các model được chỉ định.
        
        Args:
            model_names: Danh sách tên model ['mobilenet', 'resnet50', 'inception', 'vgg16']
        """
        if model_names is None:
            model_names = ['mobilenet', 'resnet50']  # Mặc định 2 model cho tốc độ
        
        self.models = {}
        self.model_transforms = {}
        
        print("🔧 Đang khởi tạo Ensemble Models (H2 - Transferability)...")
        
        for name in model_names:
            print(f"   📦 Loading {name}...", end=" ")
            model, transform = self._load_model(name)
            if model is not None:
                self.models[name] = model
                self.model_transforms[name] = transform
                print("✅")
            else:
                print("❌ Không hỗ trợ")
        
        print(f"   ✅ Đã load {len(self.models)} models: {list(self.models.keys())}")
    
    def _load_model(self, name: str):
        """Load model theo tên"""
        try:
            if name == 'mobilenet':
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                transform = preprocess
            elif name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                transform = preprocess
            elif name == 'inception':
                model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
                model.aux_logits = False
                # Inception cần 299x299
                transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            elif name == 'vgg16':
                model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                transform = preprocess
            elif name == 'densenet':
                model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                transform = preprocess
            else:
                return None, None
            
            model = model.to(DEVICE)
            model.eval()
            return model, transform
        except Exception as e:
            print(f"Lỗi load {name}: {e}")
            return None, None
    
    def get_ensemble_prediction(self, image_tensor: torch.Tensor) -> dict:
        """
        Lấy dự đoán từ tất cả các model trong ensemble.
        
        Returns:
            dict: {model_name: (predicted_class, confidence)}
        """
        results = {}
        with torch.no_grad():
            for name, model in self.models.items():
                # Resize nếu cần (Inception cần 299x299)
                if name == 'inception' and image_tensor.shape[-1] != 299:
                    img = F.interpolate(image_tensor, size=(299, 299), mode='bilinear')
                else:
                    img = image_tensor
                
                output = model(img)
                probs = F.softmax(output, dim=1)
                conf, pred = probs.max(1)
                results[name] = (pred.item(), conf.item())
        
        return results
    
    def compute_ensemble_fitness(self, image_tensor: torch.Tensor, 
                                  original_labels: dict = None) -> float:
        """
        Tính điểm fitness cho ensemble attack.
        Điểm cao = nhiều model bị đánh lừa.
        
        Args:
            image_tensor: Ảnh đã được áp dụng patch
            original_labels: Dict {model_name: original_label} để so sánh
        
        Returns:
            float: Điểm fitness (0-1, 1 = tất cả model bị lừa)
        """
        predictions = self.get_ensemble_prediction(image_tensor)
        
        if original_labels is None:
            # Nếu không có label gốc, tính dựa trên độ tin cậy thấp
            total_uncertainty = 0
            for name, (pred, conf) in predictions.items():
                total_uncertainty += (1 - conf)  # Uncertainty càng cao càng tốt
            return total_uncertainty / len(predictions)
        else:
            # Tính số model bị đánh lừa
            fooled = 0
            for name, (pred, conf) in predictions.items():
                if name in original_labels and pred != original_labels[name]:
                    fooled += 1
            return fooled / len(predictions)


# ==========================================
# PHẦN 3: H3 - SEMANTIC PATCH REPRESENTATION
# ==========================================

@dataclass
class Circle:
    """Đại diện cho một hình tròn trong patch (Gen trong GA)"""
    x: float        # Vị trí x (0-1, tỉ lệ với kích thước patch)
    y: float        # Vị trí y (0-1)
    radius: float   # Bán kính (0-0.5)
    r: int          # Red (0-255)
    g: int          # Green (0-255)
    b: int          # Blue (0-255)
    alpha: int      # Độ trong suốt (0-255)
    
    def mutate(self, mutation_rate: float = 0.1):
        """Đột biến ngẫu nhiên các thuộc tính"""
        if random.random() < mutation_rate:
            self.x = np.clip(self.x + random.gauss(0, 0.1), 0, 1)
        if random.random() < mutation_rate:
            self.y = np.clip(self.y + random.gauss(0, 0.1), 0, 1)
        if random.random() < mutation_rate:
            self.radius = np.clip(self.radius + random.gauss(0, 0.05), 0.02, 0.4)
        if random.random() < mutation_rate:
            self.r = int(np.clip(self.r + random.gauss(0, 30), 0, 255))
        if random.random() < mutation_rate:
            self.g = int(np.clip(self.g + random.gauss(0, 30), 0, 255))
        if random.random() < mutation_rate:
            self.b = int(np.clip(self.b + random.gauss(0, 30), 0, 255))
        if random.random() < mutation_rate:
            self.alpha = int(np.clip(self.alpha + random.gauss(0, 30), 50, 255))
    
    def copy(self):
        return Circle(self.x, self.y, self.radius, self.r, self.g, self.b, self.alpha)


@dataclass
class Rectangle:
    """Đại diện cho hình chữ nhật (thêm đa dạng hình học)"""
    x: float
    y: float
    width: float
    height: float
    r: int
    g: int
    b: int
    alpha: int
    rotation: float  # Góc xoay (độ)
    
    def mutate(self, mutation_rate: float = 0.1):
        if random.random() < mutation_rate:
            self.x = np.clip(self.x + random.gauss(0, 0.1), 0, 1)
        if random.random() < mutation_rate:
            self.y = np.clip(self.y + random.gauss(0, 0.1), 0, 1)
        if random.random() < mutation_rate:
            self.width = np.clip(self.width + random.gauss(0, 0.05), 0.05, 0.5)
        if random.random() < mutation_rate:
            self.height = np.clip(self.height + random.gauss(0, 0.05), 0.05, 0.5)
        if random.random() < mutation_rate:
            self.r = int(np.clip(self.r + random.gauss(0, 30), 0, 255))
        if random.random() < mutation_rate:
            self.g = int(np.clip(self.g + random.gauss(0, 30), 0, 255))
        if random.random() < mutation_rate:
            self.b = int(np.clip(self.b + random.gauss(0, 30), 0, 255))
        if random.random() < mutation_rate:
            self.alpha = int(np.clip(self.alpha + random.gauss(0, 30), 50, 255))
        if random.random() < mutation_rate:
            self.rotation = (self.rotation + random.gauss(0, 15)) % 360
    
    def copy(self):
        return Rectangle(self.x, self.y, self.width, self.height, 
                        self.r, self.g, self.b, self.alpha, self.rotation)


class SemanticPatch:
    """
    Miếng dán nghệ thuật được tạo từ các hình học cơ bản.
    Đây là "cá thể" trong giải thuật di truyền.
    """
    
    def __init__(self, size: int = 100, num_circles: int = 30, num_rects: int = 10):
        """
        Khởi tạo patch ngẫu nhiên.
        
        Args:
            size: Kích thước patch (pixel)
            num_circles: Số hình tròn
            num_rects: Số hình chữ nhật
        """
        self.size = size
        self.circles = [self._random_circle() for _ in range(num_circles)]
        self.rectangles = [self._random_rectangle() for _ in range(num_rects)]
        self.fitness = 0.0
        self._cached_image = None
        self._cache_valid = False
    
    def _random_circle(self) -> Circle:
        """Tạo hình tròn ngẫu nhiên"""
        return Circle(
            x=random.random(),
            y=random.random(),
            radius=random.uniform(0.05, 0.3),
            r=random.randint(0, 255),
            g=random.randint(0, 255),
            b=random.randint(0, 255),
            alpha=random.randint(100, 255)
        )
    
    def _random_rectangle(self) -> Rectangle:
        """Tạo hình chữ nhật ngẫu nhiên"""
        return Rectangle(
            x=random.random(),
            y=random.random(),
            width=random.uniform(0.1, 0.4),
            height=random.uniform(0.1, 0.4),
            r=random.randint(0, 255),
            g=random.randint(0, 255),
            b=random.randint(0, 255),
            alpha=random.randint(100, 255),
            rotation=random.uniform(0, 360)
        )
    
    def render(self) -> Image.Image:
        """
        Render patch thành ảnh PIL.
        Sử dụng cache để tăng tốc.
        """
        if self._cache_valid and self._cached_image is not None:
            return self._cached_image
        
        # Tạo ảnh RGBA (có alpha channel)
        img = Image.new('RGBA', (self.size, self.size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Vẽ rectangles trước (layer dưới)
        for rect in self.rectangles:
            x1 = int(rect.x * self.size)
            y1 = int(rect.y * self.size)
            x2 = int((rect.x + rect.width) * self.size)
            y2 = int((rect.y + rect.height) * self.size)
            draw.rectangle([x1, y1, x2, y2], 
                          fill=(rect.r, rect.g, rect.b, rect.alpha))
        
        # Vẽ circles (layer trên)
        for circle in self.circles:
            cx = int(circle.x * self.size)
            cy = int(circle.y * self.size)
            r = int(circle.radius * self.size)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                        fill=(circle.r, circle.g, circle.b, circle.alpha))
        
        self._cached_image = img
        self._cache_valid = True
        return img
    
    def invalidate_cache(self):
        """Đánh dấu cache không còn hợp lệ sau khi đột biến"""
        self._cache_valid = False
    
    def to_tensor(self) -> torch.Tensor:
        """Chuyển patch thành tensor để áp dụng vào ảnh"""
        img = self.render()
        # Chuyển RGBA sang RGB
        rgb_img = Image.new('RGB', img.size, (128, 128, 128))
        rgb_img.paste(img, mask=img.split()[3])  # Paste với alpha mask
        
        # Chuyển sang tensor và normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform(rgb_img)
    
    def mutate(self, mutation_rate: float = 0.15):
        """Đột biến patch"""
        self.invalidate_cache()
        for circle in self.circles:
            circle.mutate(mutation_rate)
        for rect in self.rectangles:
            rect.mutate(mutation_rate)
    
    def crossover(self, other: 'SemanticPatch') -> 'SemanticPatch':
        """Lai ghép với patch khác"""
        child = SemanticPatch(self.size, 0, 0)
        child.circles = []
        child.rectangles = []
        
        # Lai ghép circles
        for i in range(len(self.circles)):
            if random.random() < 0.5:
                child.circles.append(self.circles[i].copy())
            else:
                child.circles.append(other.circles[i].copy())
        
        # Lai ghép rectangles
        for i in range(len(self.rectangles)):
            if random.random() < 0.5:
                child.rectangles.append(self.rectangles[i].copy())
            else:
                child.rectangles.append(other.rectangles[i].copy())
        
        return child
    
    def copy(self) -> 'SemanticPatch':
        """Tạo bản sao"""
        new_patch = SemanticPatch(self.size, 0, 0)
        new_patch.circles = [c.copy() for c in self.circles]
        new_patch.rectangles = [r.copy() for r in self.rectangles]
        new_patch.fitness = self.fitness
        return new_patch


# ==========================================
# PHẦN 4: H4 - EOT (Expectation Over Transformation)
# ==========================================

class EOTTransformer:
    """
    Expectation Over Transformation - Giả lập các biến đổi vật lý.
    Đây là chìa khóa để patch hoạt động trong thế giới thực.
    """
    
    def __init__(self, 
                rotation_range: Tuple[float, float] = (-30, 30),
                scale_range: Tuple[float, float] = (0.8, 1.2),
                brightness_range: Tuple[float, float] = (0.7, 1.3),
                noise_level: float = 0.05,
                blur_prob: float = 0.3,
                perspective_strength: float = 0.1):
        """
        Khởi tạo EOT với các tham số biến đổi.
        
        Args:
            rotation_range: Khoảng góc xoay (độ)
            scale_range: Khoảng scale (tỉ lệ)
            brightness_range: Khoảng độ sáng
            noise_level: Mức nhiễu Gaussian
            blur_prob: Xác suất bị blur
            perspective_strength: Độ mạnh biến đổi perspective
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.blur_prob = blur_prob
        self.perspective_strength = perspective_strength
    
    def random_rotation(self, image: Image.Image) -> Image.Image:
        """Xoay ảnh ngẫu nhiên"""
        angle = random.uniform(*self.rotation_range)
        return image.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    def random_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scale tensor ngẫu nhiên"""
        scale = random.uniform(*self.scale_range)
        h, w = tensor.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        return F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), 
                            mode='bilinear', align_corners=False).squeeze(0)
    
    def random_brightness(self, tensor: torch.Tensor) -> torch.Tensor:
        """Thay đổi độ sáng ngẫu nhiên"""
        factor = random.uniform(*self.brightness_range)
        return torch.clamp(tensor * factor, -3, 3)  # Clamp trong không gian normalized
    
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Thêm nhiễu Gaussian"""
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise
    
    def random_blur(self, image: Image.Image) -> Image.Image:
        """Áp dụng blur ngẫu nhiên"""
        if random.random() < self.blur_prob:
            radius = random.uniform(0.5, 2.0)
            return image.filter(ImageFilter.GaussianBlur(radius))
        return image
    
    def apply_random_transform(self, patch_image: Image.Image) -> Image.Image:
        """Áp dụng chuỗi biến đổi ngẫu nhiên cho patch (trên PIL Image)"""
        # Rotation
        transformed = self.random_rotation(patch_image)
        
        # Blur (simulate camera blur)
        transformed = self.random_blur(transformed)
        
        return transformed
    
    def apply_tensor_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        """Áp dụng biến đổi trên tensor (sau khi patch đã được áp dụng)"""
        # Brightness
        tensor = self.random_brightness(tensor)
        
        # Noise
        tensor = self.add_noise(tensor)
        
        return tensor


# ==========================================
# PHẦN 5: PATCH APPLICATOR
# ==========================================

class PatchApplicator:
    """
    Áp dụng patch vào ảnh tại vị trí chỉ định.
    """
    
    def __init__(self, patch_size: int = 80, image_size: int = 224):
        self.patch_size = patch_size
        self.image_size = image_size
    
    def apply_patch(self, 
                    image_tensor: torch.Tensor,
                    patch: SemanticPatch,
                    position: Tuple[int, int] = None,
                    eot_transformer: EOTTransformer = None) -> torch.Tensor:
        """
        Áp dụng patch vào ảnh.
        
        Args:
            image_tensor: Ảnh gốc (C, H, W) đã normalize
            patch: SemanticPatch object
            position: (x, y) vị trí đặt patch, None = random
            eot_transformer: EOT transformer để áp dụng biến đổi
        
        Returns:
            Ảnh đã được áp dụng patch
        """
        result = image_tensor.clone()
        
        # Render patch
        patch_img = patch.render()
        
        # Áp dụng EOT transforms nếu có
        if eot_transformer is not None:
            patch_img = eot_transformer.apply_random_transform(patch_img)
        
        # Resize patch về kích thước mong muốn
        patch_img = patch_img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        
        # Chuyển sang tensor
        patch_rgb = Image.new('RGB', patch_img.size, (128, 128, 128))
        patch_rgb.paste(patch_img, mask=patch_img.split()[3])
        
        patch_tensor = transforms.ToTensor()(patch_rgb)
        # Normalize
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        patch_tensor = (patch_tensor - mean) / std
        patch_tensor = patch_tensor.to(image_tensor.device)
        
        # Tạo mask từ alpha channel
        alpha = transforms.ToTensor()(patch_img)[3]  # Alpha channel
        mask = (alpha > 0.1).float().to(image_tensor.device)
        mask = mask.unsqueeze(0).expand(3, -1, -1)
        
        # Vị trí đặt patch
        if position is None:
            max_x = self.image_size - self.patch_size
            max_y = self.image_size - self.patch_size
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            x, y = position
            x = min(x, self.image_size - self.patch_size)
            y = min(y, self.image_size - self.patch_size)
        
        # Áp dụng patch với alpha blending
        result[:, y:y+self.patch_size, x:x+self.patch_size] = \
            result[:, y:y+self.patch_size, x:x+self.patch_size] * (1 - mask) + \
            patch_tensor * mask
        
        # Áp dụng EOT tensor transforms nếu có
        if eot_transformer is not None:
            result = eot_transformer.apply_tensor_transforms(result)
        
        return result


# ==========================================
# PHẦN 6: H1 - GENETIC ALGORITHM OPTIMIZER
# ==========================================

class GeneticAlgorithmOptimizer:
    """
    Giải thuật di truyền để tối ưu hóa patch (Black-box - H1).
    """
    
    def __init__(self,
                population_size: int = 50,
                patch_size: int = 80,
                num_circles: int = 25,
                num_rects: int = 8,
                mutation_rate: float = 0.15,
                elite_ratio: float = 0.1,
                tournament_size: int = 5):
        """
        Khởi tạo GA Optimizer.
        
        Args:
            population_size: Kích thước quần thể
            patch_size: Kích thước patch (pixel)
            num_circles: Số hình tròn trong mỗi patch
            num_rects: Số hình chữ nhật
            mutation_rate: Tỉ lệ đột biến
            elite_ratio: Tỉ lệ elite (giữ nguyên)
            tournament_size: Kích thước tournament selection
        """
        self.population_size = population_size
        self.patch_size = patch_size
        self.num_circles = num_circles
        self.num_rects = num_rects
        self.mutation_rate = mutation_rate
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.tournament_size = tournament_size
        
        # Khởi tạo quần thể
        self.population = [
            SemanticPatch(patch_size, num_circles, num_rects) 
            for _ in range(population_size)
        ]
        
        self.best_patch = None
        self.best_fitness = 0
        self.generation = 0
        self.fitness_history = []
    
    def tournament_selection(self) -> SemanticPatch:
        """Chọn lọc theo tournament"""
        tournament = random.sample(self.population, self.tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        return winner
    
    def evolve(self):
        """Tiến hóa một thế hệ"""
        # Sắp xếp theo fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Lưu best
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_patch = self.population[0].copy()
        
        # Elite selection (giữ nguyên top performers)
        new_population = [p.copy() for p in self.population[:self.elite_count]]
        
        # Tạo phần còn lại bằng crossover và mutation
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1


# ==========================================
# PHẦN 7: MAIN ATTACK FRAMEWORK
# ==========================================

class UniversalGhostPatch:
    """
    Framework chính tích hợp tất cả components.
    """
    
    def __init__(self,
                 model_names: List[str] = None,
                 population_size: int = 30,
                 patch_size: int = 80,
                 num_circles: int = 25,
                 num_rects: int = 8,
                 use_eot: bool = True):
        """
        Khởi tạo Universal Ghost Patch framework.
        
        Args:
            model_names: Danh sách model cho ensemble
            population_size: Kích thước quần thể GA
            patch_size: Kích thước patch
            num_circles: Số hình tròn
            num_rects: Số hình chữ nhật
            use_eot: Có sử dụng EOT hay không
        """
        print("=" * 60)
        print("🔮 UNIVERSAL GHOST PATCH - Adversarial Attack Framework")
        print("=" * 60)
        
        # H2: Ensemble Models
        if model_names is None:
            model_names = ['mobilenet', 'resnet50']
        self.ensemble = EnsembleModels(model_names)
        
        # H1: GA Optimizer
        self.optimizer = GeneticAlgorithmOptimizer(
            population_size=population_size,
            patch_size=patch_size,
            num_circles=num_circles,
            num_rects=num_rects
        )
        
        # H4: EOT Transformer
        self.use_eot = use_eot
        if use_eot:
            self.eot = EOTTransformer(
                rotation_range=(-20, 20),
                scale_range=(0.9, 1.1),
                brightness_range=(0.8, 1.2),
                noise_level=0.03,
                blur_prob=0.2
            )
        else:
            self.eot = None
        
        # Patch Applicator
        self.applicator = PatchApplicator(patch_size=patch_size)
        
        print(f"✅ Framework initialized!")
        print(f"   - Population size: {population_size}")
        print(f"   - Patch size: {patch_size}x{patch_size}")
        print(f"   - EOT enabled: {use_eot}")
        print("=" * 60)
    
    def evaluate_population(self, 
                           base_image: torch.Tensor,
                           original_labels: dict,
                           num_eot_samples: int = 3):
        """
        Đánh giá fitness của toàn bộ quần thể.
        
        Args:
            base_image: Ảnh gốc để áp dụng patch
            original_labels: Labels gốc của ảnh
            num_eot_samples: Số mẫu EOT để trung bình
        """
        for patch in self.optimizer.population:
            total_fitness = 0
            
            for _ in range(num_eot_samples):
                # Áp dụng patch với EOT
                patched_image = self.applicator.apply_patch(
                    base_image, patch, position=None, eot_transformer=self.eot
                )
                patched_image = patched_image.unsqueeze(0).to(DEVICE)
                
                # Tính fitness từ ensemble
                fitness = self.ensemble.compute_ensemble_fitness(
                    patched_image, original_labels
                )
                total_fitness += fitness
            
            patch.fitness = total_fitness / num_eot_samples
    
    def train(self,
              base_image: torch.Tensor,
              num_generations: int = 100,
              num_eot_samples: int = 3,
              verbose: bool = True,
              save_interval: int = 20):
        """
        Huấn luyện để tìm patch tối ưu.
        
        Args:
            base_image: Ảnh gốc
            num_generations: Số thế hệ
            num_eot_samples: Số mẫu EOT
            verbose: Hiển thị tiến trình
            save_interval: Khoảng cách lưu checkpoint
        
        Returns:
            SemanticPatch: Patch tốt nhất
        """
        print("\n🚀 Bắt đầu huấn luyện Universal Ghost Patch...")
        
        # Lấy label gốc từ ensemble
        base_image_batch = base_image.unsqueeze(0).to(DEVICE)
        predictions = self.ensemble.get_ensemble_prediction(base_image_batch)
        original_labels = {name: pred[0] for name, pred in predictions.items()}
        
        print(f"📋 Original predictions:")
        for name, (pred, conf) in predictions.items():
            print(f"   - {name}: {LABELS[pred]} ({conf:.2%})")
        
        print(f"\n🧬 Tiến hóa {num_generations} thế hệ...")
        
        for gen in range(num_generations):
            # Đánh giá quần thể
            self.evaluate_population(base_image, original_labels, num_eot_samples)
            
            # Lưu fitness history
            self.optimizer.fitness_history.append(self.optimizer.best_fitness)
            
            if verbose and gen % 10 == 0:
                print(f"   Gen {gen:4d}: Best fitness = {self.optimizer.best_fitness:.4f}")
            
            # Tiến hóa
            self.optimizer.evolve()
            
            # Lưu checkpoint
            if gen % save_interval == 0 and self.optimizer.best_patch is not None:
                self._save_checkpoint(gen)
        
        # Đánh giá lần cuối
        self.evaluate_population(base_image, original_labels, num_eot_samples)
        
        print(f"\n✅ Huấn luyện hoàn tất!")
        print(f"   Best fitness: {self.optimizer.best_fitness:.4f}")
        
        return self.optimizer.best_patch
    
    def _save_checkpoint(self, generation: int):
        """Lưu checkpoint patch"""
        if self.optimizer.best_patch is not None:
            patch_img = self.optimizer.best_patch.render()
            filename = f"checkpoint_gen{generation}.png"
            patch_img.save(filename)
    
    def evaluate_final(self, base_image: torch.Tensor, patch: SemanticPatch):
        """
        Đánh giá patch cuối cùng.
        """
        print("\n📊 Đánh giá cuối cùng:")
        
        # Áp dụng patch (không có EOT để so sánh clean)
        patched_image = self.applicator.apply_patch(
            base_image, patch, position=(70, 70), eot_transformer=None
        )
        patched_image_batch = patched_image.unsqueeze(0).to(DEVICE)
        
        # Lấy predictions
        original_batch = base_image.unsqueeze(0).to(DEVICE)
        orig_preds = self.ensemble.get_ensemble_prediction(original_batch)
        adv_preds = self.ensemble.get_ensemble_prediction(patched_image_batch)
        
        print("\n   Model          | Original          | After Patch")
        print("   " + "-" * 55)
        
        fooled_count = 0
        for name in orig_preds.keys():
            orig_label, orig_conf = orig_preds[name]
            adv_label, adv_conf = adv_preds[name]
            
            fooled = "✅ FOOLED" if orig_label != adv_label else "❌"
            if orig_label != adv_label:
                fooled_count += 1
            
            print(f"   {name:14s} | {LABELS[orig_label][:15]:15s} | {LABELS[adv_label][:15]:15s} {fooled}")
        
        print(f"\n   📈 Attack Success Rate: {fooled_count}/{len(orig_preds)} models ({fooled_count/len(orig_preds):.1%})")
        
        return patched_image
    
    def visualize(self, original: torch.Tensor, patched: torch.Tensor, patch: SemanticPatch):
        """Hiển thị kết quả"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        orig_img = denormalize(original.cpu()).permute(1, 2, 0).numpy()
        axes[0].imshow(orig_img)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        # Patch
        patch_img = patch.render()
        axes[1].imshow(patch_img)
        axes[1].set_title("Adversarial Patch (H3 - Semantic)", fontsize=12)
        axes[1].axis('off')
        
        # Patched image
        patched_img = denormalize(patched.cpu()).permute(1, 2, 0).numpy()
        axes[2].imshow(patched_img)
        axes[2].set_title("Patched Image", fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("result_visualization.png", dpi=150)
        plt.show()
    
    def save_patch(self, patch: SemanticPatch, filename: str = "adversarial_patch.png"):
        """
        Lưu patch để in ra (H4 - Physical Attack).
        """
        # Render với kích thước lớn hơn để in
        original_size = patch.size
        patch.size = 500  # Resize lên để in rõ hơn
        patch.invalidate_cache()
        
        high_res_patch = patch.render()
        high_res_patch.save(filename)
        
        # Restore
        patch.size = original_size
        patch.invalidate_cache()
        
        print(f"💾 Đã lưu patch: {filename}")
        print(f"   📏 Kích thước: 500x500 pixels")
        print(f"   🖨️  Có thể in ra giấy để test Physical Attack (H4)")


# ==========================================
# PHẦN 8: WHITE-BOX PATCH ATTACK (Nhanh hơn)
# ==========================================

class WhiteBoxPatchAttack:
    """
    Tấn công White-box để tạo patch (nhanh hơn GA).
    Sử dụng PGD trực tiếp trên pixel của patch.
    Đây là bước "ăn chắc" - làm trước khi chuyển sang Black-box.
    """
    
    def __init__(self, 
                 model_names: List[str] = None,
                 patch_size: int = 80,
                 image_size: int = 224):
        
        print("=" * 60)
        print("⚡ WHITE-BOX PATCH ATTACK (Fast Mode)")
        print("=" * 60)
        
        if model_names is None:
            model_names = ['mobilenet', 'resnet50']
        
        self.ensemble = EnsembleModels(model_names)
        self.patch_size = patch_size
        self.image_size = image_size
        self.eot = EOTTransformer()
    
    def create_patch(self,
                     base_image: torch.Tensor,
                     num_steps: int = 200,
                     lr: float = 0.01,
                     num_eot_samples: int = 5) -> torch.Tensor:
        """
        Tạo patch sử dụng gradient descent.
        
        Args:
            base_image: Ảnh gốc
            num_steps: Số bước tối ưu
            lr: Learning rate
            num_eot_samples: Số mẫu EOT
        
        Returns:
            Patch tensor tối ưu
        """
        print(f"\n🚀 Tạo patch bằng White-box PGD ({num_steps} steps)...")
        
        # Khởi tạo patch ngẫu nhiên
        patch = torch.randn(3, self.patch_size, self.patch_size, 
                           device=DEVICE, requires_grad=True)
        
        optimizer = torch.optim.Adam([patch], lr=lr)
        
        # Lấy original labels
        base_batch = base_image.unsqueeze(0).to(DEVICE)
        orig_preds = self.ensemble.get_ensemble_prediction(base_batch)
        original_labels = {name: torch.tensor([pred[0]], device=DEVICE) 
                         for name, pred in orig_preds.items()}
        
        for step in range(num_steps):
            optimizer.zero_grad()
            total_loss = 0
            
            for _ in range(num_eot_samples):
                # Apply patch to random position
                x = random.randint(0, self.image_size - self.patch_size)
                y = random.randint(0, self.image_size - self.patch_size)
                
                patched = base_image.clone().to(DEVICE)
                patched[:, y:y+self.patch_size, x:x+self.patch_size] = patch
                patched = patched.unsqueeze(0)
                
                # EOT transforms
                patched = self.eot.apply_tensor_transforms(patched.squeeze(0)).unsqueeze(0)
                
                # Compute loss for each model
                for name, model in self.ensemble.models.items():
                    if name == 'inception':
                        img = F.interpolate(patched, size=(299, 299), mode='bilinear')
                    else:
                        img = patched
                    
                    output = model(img)
                    # Minimize probability of correct class
                    loss = -F.cross_entropy(output, original_labels[name])
                    total_loss += loss
            
            total_loss = total_loss / (num_eot_samples * len(self.ensemble.models))
            total_loss.backward()
            optimizer.step()
            
            # Clamp patch to valid range
            with torch.no_grad():
                patch.clamp_(-3, 3)
            
            if step % 20 == 0:
                print(f"   Step {step:4d}: Loss = {total_loss.item():.4f}")
        
        print("✅ White-box patch created!")
        return patch.detach()
    
    def save_patch_as_image(self, patch_tensor: torch.Tensor, filename: str = "whitebox_patch.png"):
        """Lưu patch tensor thành ảnh"""
        # Denormalize
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(patch_tensor.device)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(patch_tensor.device)
        patch_img = patch_tensor * std + mean
        patch_img = torch.clamp(patch_img, 0, 1)
        
        # Convert to PIL
        patch_np = (patch_img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(patch_np).save(filename)
        print(f"💾 Saved: {filename}")


# ==========================================
# PHẦN 9: DEMO & TESTING
# ==========================================

def demo_black_box_attack():
    """Demo tấn công Black-box với GA"""
    
    print("\n" + "=" * 60)
    print("🎮 DEMO: BLACK-BOX ATTACK (Genetic Algorithm)")
    print("=" * 60)
    
    # Tải ảnh test
    img_url = "https://i.imgur.com/Dvkyrua.jpeg"
    print(f"📷 Loading test image...")
    
    try:
        response = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        orig_image = Image.open(BytesIO(response.content)).convert('RGB')
    except:
        print("⚠️  Cannot load remote image. Creating synthetic test image...")
        orig_image = Image.new('RGB', (224, 224), color='lightblue')
    
    # Preprocess
    input_tensor = preprocess(orig_image).to(DEVICE)
    
    # Khởi tạo framework
    framework = UniversalGhostPatch(
        model_names=['mobilenet', 'resnet50'],
        population_size=20,  # Giảm để demo nhanh
        patch_size=70,
        num_circles=20,
        num_rects=5,
        use_eot=True
    )
    
    # Train
    best_patch = framework.train(
        base_image=input_tensor,
        num_generations=30,  # Giảm để demo nhanh
        num_eot_samples=2,
        verbose=True
    )
    
    # Evaluate
    patched_image = framework.evaluate_final(input_tensor, best_patch)
    
    # Visualize
    framework.visualize(input_tensor, patched_image, best_patch)
    
    # Save
    framework.save_patch(best_patch, "demo_adversarial_patch.png")
    
    return best_patch


def demo_white_box_attack():
    """Demo tấn công White-box (nhanh hơn)"""
    
    print("\n" + "=" * 60)
    print("⚡ DEMO: WHITE-BOX ATTACK (PGD - Fast)")
    print("=" * 60)
    
    # Tải ảnh test
    img_url = "https://i.imgur.com/Dvkyrua.jpeg"
    print(f"📷 Loading test image...")
    
    try:
        response = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        orig_image = Image.open(BytesIO(response.content)).convert('RGB')
    except:
        print("⚠️  Cannot load remote image. Creating synthetic test image...")
        orig_image = Image.new('RGB', (224, 224), color='lightblue')
    
    # Preprocess
    input_tensor = preprocess(orig_image).to(DEVICE)
    
    # Khởi tạo
    attacker = WhiteBoxPatchAttack(
        model_names=['mobilenet', 'resnet50'],
        patch_size=80
    )
    
    # Tạo patch
    patch = attacker.create_patch(
        base_image=input_tensor,
        num_steps=100,  # Giảm để demo nhanh
        num_eot_samples=3
    )
    
    # Save
    attacker.save_patch_as_image(patch, "whitebox_demo_patch.png")
    
    return patch


def quick_test():
    """Test nhanh các component"""
    
    print("\n" + "=" * 60)
    print("🧪 QUICK TEST - Kiểm tra các component")
    print("=" * 60)
    
    # Test SemanticPatch
    print("\n1️⃣  Testing SemanticPatch (H3)...")
    patch = SemanticPatch(size=100, num_circles=15, num_rects=5)
    patch_img = patch.render()
    patch_img.save("test_semantic_patch.png")
    print(f"   ✅ Created patch with {len(patch.circles)} circles, {len(patch.rectangles)} rects")
    
    # Test EOT
    print("\n2️⃣  Testing EOT Transformer (H4)...")
    eot = EOTTransformer()
    transformed = eot.apply_random_transform(patch_img)
    transformed.save("test_eot_transform.png")
    print("   ✅ EOT transforms applied")
    
    # Test Ensemble
    print("\n3️⃣  Testing Ensemble Models (H2)...")
    ensemble = EnsembleModels(['mobilenet'])
    
    # Create dummy image
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    preds = ensemble.get_ensemble_prediction(dummy)
    print(f"   ✅ Ensemble prediction: {preds}")
    
    # Test GA
    print("\n4️⃣  Testing GA Optimizer (H1)...")
    ga = GeneticAlgorithmOptimizer(population_size=10, patch_size=80)
    print(f"   ✅ Initialized population of {len(ga.population)} patches")
    
    print("\n✅ All components working!")
    print("=" * 60)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     UNIVERSAL GHOST PATCH - Adversarial Attack Framework       ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Chọn chế độ chạy:                                             ║
    ║    1. quick_test()     - Kiểm tra nhanh các component          ║
    ║    2. demo_white_box() - Demo White-box attack (nhanh)         ║
    ║    3. demo_black_box() - Demo Black-box GA attack (chậm hơn)   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Mặc định chạy quick test
    quick_test()
    
    # Uncomment để chạy demo:
    # demo_white_box_attack()
    # demo_black_box_attack()
