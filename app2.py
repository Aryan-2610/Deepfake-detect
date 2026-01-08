import torch
from torchvision import transforms
from PIL import Image
from model_arch import SimpleEfficientTransformer

# -----------------------------
# Load model
# -----------------------------
def load_model(path, device):
    model = SimpleEfficientTransformer()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = "/Users/aryangupta/college2/projects/deepfake-detect/efficient_transformer.pth"
    IMAGE_PATH = "image6.png"  

    model = load_model(MODEL_PATH, DEVICE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(IMAGE_PATH).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()

    label = "Deepfake: NO" if prediction < 0.535 else "Deepfake: YES"

    print("Image Path :", IMAGE_PATH)
    print("Prediction :", label)
    print("Confidence :", round(prediction, 4))


if __name__ == "__main__":
    main()
