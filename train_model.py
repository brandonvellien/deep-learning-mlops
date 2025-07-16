import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 1. Configuration du Dataset ---

path_to_dataset = '/Users/brandon/Documents/WCS/deep-learning-mlops/data/PetImages' 

# Vérification de l'existence du chemin
if not os.path.exists(path_to_dataset):
    print(f"Erreur: Le chemin du dataset '{path_to_dataset}' n'existe pas.")
    print("Veuillez vérifier que le dossier 'data' contient bien vos sous-dossiers 'Cat' et 'Dog' (ou 'cats'/'dogs').")
    print("Si votre dataset est dans 'data/PetImages', changez path_to_dataset = 'data/PetImages'")
    exit()

# Définition des classes (labels)
CLASSES = ['Cat', 'Dog'] 

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = len(CLASSES)
EPOCHS = 2 
LEARNING_RATE = 0.0001

print(f"Chargement des images depuis: {path_to_dataset}")

# hargement et l'augmentation des données

datagen = ImageDataGenerator(
    rescale=1./255, # Normalise les pixels entre 0 et 1
    rotation_range=20, # Augmentation de données (rotation aléatoire)
    width_shift_range=0.2, # Décalage horizontal aléatoire
    height_shift_range=0.2, # Décalage vertical aléatoire
    shear_range=0.2, # Cisaillage aléatoire
    zoom_range=0.2, # Zoom aléatoire
    horizontal_flip=True, # Retournement horizontal aléatoire
    fill_mode='nearest', # Remplissage des pixels manquants
    validation_split=0.2 # 20% des données pour la validation
)

train_generator = datagen.flow_from_directory(
    path_to_dataset,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Pour la classification multiclasse
    subset='training',
    classes=CLASSES
)

validation_generator = datagen.flow_from_directory(
    path_to_dataset,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=CLASSES
)

if train_generator.samples == 0 or validation_generator.samples == 0:
    print("Erreur: Aucun échantillon trouvé dans les générateurs d'entraînement ou de validation.")
    print("Vérifiez le chemin du dataset et la structure des sous-dossiers des classes (ex: data/Cat, data/Dog).")
    exit()

print(f"Total d'images d'entraînement: {train_generator.samples}")
print(f"Total d'images de validation: {validation_generator.samples}")
print(f"Classes détectées: {train_generator.class_indices}")


#Chargement du Modèle Pré-entraîné (MobileNetV2)
print("Chargement du modèle MobileNetV2 pré-entraîné...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

#Ajout de nouvelles couches pour la classification binaire 

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(NUM_CLASSES, activation='softmax')(x) #

model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

#Compilation du modèle

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

print("Modèle compilé. Démarrage de l'entraînement...")

#zntraînement du modèle ---

callbacks = [
    EarlyStopping(patience=3, monitor='val_loss', mode='min', restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

print("Entraînement terminé.")

#évaluation des performances
print("Évaluation des performances finales sur l'ensemble de validation...")
loss, accuracy = model.evaluate(validation_generator)
print(f"Perte de validation finale: {loss:.4f}")
print(f"Précision de validation finale: {accuracy:.4f}")

#sauvegarde du modèle entraîné

model_save_path = 'cats_dogs_classifier.h5'
model.save(model_save_path)
print(f"Modèle sauvegardé au format H5 : {model_save_path}")

#Affichage des courbes d'entraînement (pour le rapport technique)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision d\'entraînement')
plt.plot(history.history['val_accuracy'], label='Précision de validation')
plt.title('Précision du Modèle')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.title('Perte du Modèle')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show() 