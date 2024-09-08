from fastapi import FastAPI, Depends, UploadFile, File
from pydantic import BaseModel  # this is a super class for anythng that is strictly typed in python and has nothing to do with machine learning
from torchvision import transforms
from torchvision.models import ResNet
# This is the python imaging library that is used to read images
from PIL import Image
import io
import torch
import torch.nn.functional as F

from app.model import load_model, load_transforms, CATEGORIES

# this is what we use the BaseModel for
# The result is strictly typed so that it retuns the predicted category 
# and the confidence of the prediction
class Result(BaseModel):
    category: str
    confidence: float


app = FastAPI()


@app.post("/predict", response_model=Result)

# this is to make sure that multiple users can use the model at the same time asynchrounously
async def predict(
        input_image: UploadFile = File(...),
        model: ResNet = Depends(load_model),
        transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    # Read the uploaded image
    image = Image.open(io.BytesIO(await input_image.read()))

    # Convert RGBA image to RGB image
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply the transformations
    image = transforms(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    # Map the predicted class index to the category
    category = CATEGORIES[predicted_class.item()]

    return Result(category=category, confidence=confidence.item())