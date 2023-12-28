import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from io import BytesIO

st.set_page_config(page_title = "Mask Classification", page_icon=":mask:", layout = "wide", initial_sidebar_state = "expanded")

st.title(":mag: Mask Classification")
st.write("Upload an image! We'll analyze it to determine the presence of a mask, as well as estimate age and gender.:thinking_face:")
st.markdown("<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #635985;'><br>", True)

st.sidebar.write("## :camera_with_flash: Upload an image")

MAX_FILE_SIZE = 5 * 1024 * 1024

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.mask = nn.Linear(1000, 3, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)

    def forward(self, x):
        x = self.resnet(x)
        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)
        return mask, gender, age


model_path = "./best.pth"
num_classes = 18
model = ResNet50(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.548,0.504,0.479], std=[0.237,0.247,0.246])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def convert_iamge(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload):
    # image = Image.open(upload)
    col1.subheader("🖼️Your image")
    col1.image(upload)

    col2.subheader("🕵️‍♀️Your mask, gender, age")

    input_image = preprocess_image(upload)
    with torch.no_grad():
        mask, gender, age = model(input_image)
        mask_out = mask.argmax(dim=-1)
        gender_out = gender.argmax(dim=-1)
        age_out = age.argmax(dim=-1)

    if mask_out == 0:
        mask_out = "😷Wear!"
    elif mask_out == 1:
        mask_out = "🤭Incorrect!"
    elif mask_out == 2:
        mask_out = "😊Not wear!"
    
    if gender_out == 0:
        gender_out = "👨Male!"
    elif gender_out == 1:
        gender_out = "👩Female!"
    
    if age_out == 0:
        age_out = "🧒Young(~29)!"
    elif age_out == 1:
        age_out = "🧑‍🦰Middle(30~59)!"
    elif age_out == 2:
        age_out = "🧓Old(60~)!"

    col2.write(f":white_check_mark: Mask: {mask_out}")
    col2.write(f":white_check_mark: Gender: {gender_out}")
    col2.write(f":white_check_mark: Age: {age_out}")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpge"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smallar than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("./mask_child.jpg")
