
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load pre-trained ResNet model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the image for the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Predict recyclable items
def predict_recyclable(image, model):
    # Preprocess the image
    img_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map predicted class index to label
    recyclable_classes = [
        'bottle', 'can', 'paper', 'cardboard', 'glass', 'metal', 'plastic', 'envelope', 'newspaper', 'magazine'
    ]
    predicted_label = recyclable_classes[predicted.item() % len(recyclable_classes)]

    # Check if the predicted label is recyclable
    recyclable_keywords = [
        'bottle', 'can', 'paper', 'cardboard', 'glass', 'metal', 'plastic', 'envelope', 'newspaper', 'magazine'
    ]
    is_recyclable = any(keyword in predicted_label.lower() for keyword in recyclable_keywords)

    return is_recyclable, predicted_label

# Streamlit app
st.title("♻️ Recyclable Item Classifier")
st.write("Upload an image of an item to check if it's recyclable")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = load_model()

    if st.button('Analyze'):
        with st.spinner('Checking recyclability...'):
            is_recyclable, predicted_label = predict_recyclable(image, model)

            if is_recyclable:
                st.success(f"✅ This item is likely recyclable! Predicted: {predicted_label}")
            else:
                st.error(f"❌ This item is probably not recyclable. Predicted: {predicted_label}")

st.markdown("""
            * Note: *This is a demonstration using a pre - trained ResNet model. For accurate recycling classification, a custom - trained model on specific recycling datasets is recommended.
""")