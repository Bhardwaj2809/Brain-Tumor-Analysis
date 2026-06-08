import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# 1. Watershed Segmentation
# -----------------------------
def watershed_segmentation(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Distance transform
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # Foreground
    _, fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    fg = np.uint8(fg)

    # Background
    bg = cv2.dilate(thresh, None, iterations=3)

    # Unknown region
    unknown = cv2.subtract(bg, fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255

    return img, mask


# -----------------------------
# 2. CNN Model
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 32 * 32 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # tumor / no tumor
        )

    def forward(self, x, mask_features):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        x = torch.cat((x, mask_features), dim=1)
        x = self.fc(x)
        return x


# -----------------------------
# 3. Feature Extraction from Mask
# -----------------------------
def extract_mask_features(mask):
    mask = cv2.resize(mask, (16, 16))
    mask = mask.flatten() / 255.0
    mask = np.pad(mask, (0, 256 - len(mask)))  # ensure size = 256
    return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


# -----------------------------
# 4. Prediction Pipeline
# -----------------------------
def predict(image_path, model):
    img, mask = watershed_segmentation(image_path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)

    mask_features = extract_mask_features(mask)

    output = model(image, mask_features)
    _, predicted = torch.max(output, 1)

    return predicted.item(), mask


# -----------------------------
# 5. Run Example
# -----------------------------
if __name__ == "__main__":  
    model = CNNModel()

    # (In real case, load trained weights)
    # model.load_state_dict(torch.load("model.pth"))

    image_path = "brain_mri_01.png"

    result, mask = predict(image_path, model)

    if result == 1:
        print("Tumor Detected")
    else:
        print("No Tumor Detected")

    cv2.imshow("Watershed Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
