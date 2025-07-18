# app.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model_arch import SimpleEfficientTransformer


def load_model(path, device):
    model = SimpleEfficientTransformer()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model("/Users/aryangupta/college2/projects/deepfake-detect/efficient_transformer.pth", device=DEVICE)
MODEL.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("🧠 Deepfake Detector")
st.markdown("Upload an image and detect if it's **Deepfake: Yes or No**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = MODEL(input_tensor)
        prediction = torch.sigmoid(output).item()

    label = "Deepfake: **Yes** 🔴" if prediction <0.5 else "Deepfake: **No** 🟢"
    st.markdown(f"## Prediction: {(label,prediction)}")
