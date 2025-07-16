import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

#initialisation de l'API FastAPI
app = FastAPI(
    title="API de Classification Chats/Chiens",
    description="API pour classer les images entre chats et chiens à l'aide d'un modèle de Deep Learning."
)

#chargement du modèle au démarrage de l'API 

GLOBAL_MODEL = None
MODEL_PATH = 'cats_dogs_classifier.h5' 

@app.on_event("startup")
async def load_model():
    global GLOBAL_MODEL
    try:
        GLOBAL_MODEL = tf.keras.models.load_model(MODEL_PATH)
        GLOBAL_MODEL.summary() # Affiche un résumé du modèle pour confirmation
        print(f"Modèle '{MODEL_PATH}' chargé avec succès au démarrage de l'API.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        raise RuntimeError(f"Impossible de charger le modèle depuis {MODEL_PATH}. L'API ne démarrera pas.")

#classes et des dimensions de l'image
CLASSES = ['Cat', 'Dog']
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = len(CLASSES)

#schéma de la réponse de l'API 

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    probabilities: dict # Ajout pour les probabilités brutes

#implémentation de l'endpoint POST pour l'inférence

@app.post("/predict/", response_model=PredictionResponse, summary="Classifie une image en chat ou chien")
async def predict_image(file: UploadFile = File(...)):
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. L'API n'est pas prête.")

   
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image (JPEG, PNG, etc.).")

    try:
        
        contents = await file.read()
       
        img = Image.open(io.BytesIO(contents))

        #prétraitement de l'image pour le modèle
        
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0 

        #Faire la prédiction
        predictions = GLOBAL_MODEL.predict(img_array)
       

       
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASSES[predicted_class_index]

        
        confidence = float(predictions[0][predicted_class_index])

        
        probabilities_dict = {CLASSES[i]: float(predictions[0][i]) for i in range(NUM_CLASSES)}

        #réponse structurée
        return PredictionResponse(
            filename=file.filename,
            prediction=predicted_class_name,
            confidence=confidence,
            probabilities=probabilities_dict
        )

    except Exception as e:
        #gestion générique des erreurs lors du traitement de l'image ou de la prédiction
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors du traitement de l'image: {e}")

#point d'entrée pour le serveur 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)