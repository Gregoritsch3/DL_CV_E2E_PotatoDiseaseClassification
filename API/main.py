from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
#import requests

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

MODEL = tf.keras.models.load_model("models/model1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = np.expand_dims(image, 0)

    #json_data = {
    #    "instances": image.tolist()
    #}
    prediction = MODEL.predict(image)
    #response = requests.post(endpoint, json=json_data)
    #prediction = np.array(response.json()["prediction"][0])


    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)
