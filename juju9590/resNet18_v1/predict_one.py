from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ===== 설정 =====
MODEL_PATH = Path("best.pt")
IMG_PATH = Path(r"infer\wrongway (1).png")   # 여기에 예측할 이미지 경로 입력
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"

# 학습 때와 동일한 전처리 (eval_tf)
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# 클래스 순서: ImageFolder가 알파벳 순으로 만듦
# normal, wrongway 순서가 맞는지 train 로그에서 확인했었음
classes = ["normal", "wrongway"]

# 모델 로드 (ResNet18 + fc=2)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# 이미지 로드
img = Image.open(IMG_PATH).convert("RGB")
x = tf(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].cpu().tolist()
    pred = int(torch.argmax(logits, dim=1).item())

print(f"Image: {IMG_PATH}")
print(f"Pred: {classes[pred]}")
print(f"Prob normal={prob[0]:.3f}, wrongway={prob[1]:.3f}")
