from PIL import Image, ImageDraw
from pynput import mouse

import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# -----------------------
# CONFIG
# -----------------------
IMAGE_SIZE = 128
MODEL_PATH = "Model-Training/gesture_cnn_val_9598.pt"      # your .pt file
CLASSES_PATH = "Model-Training/gesture_classes.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Same crop function
# -----------------------
def crop_to_square_content(img, padding=50):
    img = np.array(img)

    # detect foreground (your logic: dark pixels)
    coords = np.column_stack(np.where(img < 100))

    if coords.size == 0:
        return Image.fromarray(img)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # add padding
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(img.shape[0], y_max + padding)
    x_max = min(img.shape[1], x_max + padding)

    # bounding box size
    h = y_max - y_min
    w = x_max - x_min

    # make square side = max dimension
    side = max(h, w)

    # center the square on bbox center
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2

    y1 = max(0, cy - side // 2)
    x1 = max(0, cx - side // 2)
    y2 = y1 + side
    x2 = x1 + side

    # fix if out of bounds (shift back into image)
    if y2 > img.shape[0]:
        y2 = img.shape[0]
        y1 = y2 - side
    if x2 > img.shape[1]:
        x2 = img.shape[1]
        x1 = x2 - side

    cropped = img[y1:y2, x1:x2]

    return Image.fromarray(cropped)

# -----------------------
# Validation transform
# -----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(crop_to_square_content),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.9).float())
])

# -----------------------
# Model
# -----------------------
class GestureCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# -----------------------
# Load classes
# -----------------------
with open(CLASSES_PATH) as f:
    class_names = json.load(f)

# -----------------------
# Load model
# -----------------------
model = GestureCNN(len(class_names))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# -----------------------
# Listen for mouse gestures
# -----------------------

isRecording = False
isActive = True

def on_move(x, y, injected):
    # print('Pointer moved to {}; it was {}'.format(
    #     (x, y), 'faked' if injected else 'not faked'))
    global isActive
    if x > 1900 and y < 30:
        # Stop listener
        isActive = False
        return False
    if isRecording and (x, y) not in mousePath:
        mousePath.append((x, y))

def on_click(x, y, button, pressed, injected):
    global isRecording
    # print('{} {} at {}; it was {}'.format(
    #     button,
    #     'Pressed' if pressed else 'Released',
    #     (x, y), 'faked' if injected else 'not faked'))
    if button == mouse.Button.x2 and pressed:
        isRecording = True
    elif button == mouse.Button.x2 and not pressed:
        isRecording = False
        return False  # Stop listener

def on_scroll(x, y, dx, dy, injected):
    # print('Scrolled {} at {}; it was {}'.format(
    #     'down' if dy < 0 else 'up',
    #     (x, y), 'faked' if injected else 'not faked'))
    pass

while isActive:
    mousePath = []

    print(f"Waiting for Gesture")
    with mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll) as listener:
        listener.join()
    print(f"Gesture recorded.")

    if mousePath:
        # create blank image
        img = Image.new("RGB", (1920, 1080), "white")
        draw = ImageDraw.Draw(img)

        # draw lines between points
        draw.line(mousePath, fill="black", width=3)

        img.save(f"Mouse-Gesture-Tool/gesture.png")

        # Load the image and apply the transform
        img = transform(img)
        img = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, dim=1)

        print("Prediction:", class_names[pred.item()])
        print("Confidence:", confidence.item())

        # top_probs, top_idx = torch.topk(probs, k=3)

        # for p, idx in zip(top_probs[0], top_idx[0]):
        #     print(class_names[idx.item()], p.item())