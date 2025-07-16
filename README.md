# 🚀 Projet MLOps : Classification d'Images (Chats et Chiens) et Déploiement sur AWS EC2

Ce projet a pour objectif de démontrer le cycle de vie complet d'un modèle de Deep Learning, de l'entraînement à l'inférence via une API conteneurisée, jusqu'au déploiement en production sur AWS EC2.

## 🌟 Fonctionnalités

* **Entraînement de Modèle :** Ré-entraînement (transfer learning) d'un modèle MobileNetV2 pour la classification d'images de chats et de chiens.
* **API RESTful :** Exposition du modèle entraîné via une API FastAPI, capable de recevoir une image et de renvoyer une prédiction.
* **Conteneurisation Docker :** Empaquetage de l'API et du modèle dans une image Docker pour une portabilité et une reproductibilité optimales.
* **Partage d'Image :** Publication de l'image Docker sur Docker Hub.
* **Déploiement Cloud :** Déploiement de l'application conteneurisée sur une instance Amazon EC2.

---

## 📂 Structure du Projet

Absolument ! Voici le contenu de votre README.md au format Markdown, prêt à être copié/collé dans votre fichier. Il inclut une section pour les artefacts générés (modèle H5 et historique d'entraînement) pour expliquer pourquoi ils ne sont pas versionnés, mais sont nécessaires à l'application.

Markdown

# 🚀 Projet MLOps : Classification d'Images (Chats et Chiens) et Déploiement sur AWS EC2

Ce projet a pour objectif de démontrer le cycle de vie complet d'un modèle de Deep Learning, de l'entraînement à l'inférence via une API conteneurisée, jusqu'au déploiement en production sur AWS EC2.

## 🌟 Fonctionnalités

* **Entraînement de Modèle :** Ré-entraînement (transfer learning) d'un modèle MobileNetV2 pour la classification d'images de chats et de chiens.
* **API RESTful :** Exposition du modèle entraîné via une API FastAPI, capable de recevoir une image et de renvoyer une prédiction.
* **Conteneurisation Docker :** Empaquetage de l'API et du modèle dans une image Docker pour une portabilité et une reproductibilité optimales.
* **Partage d'Image :** Publication de l'image Docker sur Docker Hub.
* **Déploiement Cloud :** Déploiement de l'application conteneurisée sur une instance Amazon EC2.

---

## 📂 Structure du Projet

.
├── app.py                  # Code source de l'API FastAPI
├── Dockerfile              # Instructions pour la construction de l'image Docker
├── requirements.txt        # Liste des dépendances Python du projet
├── train_model.py          # Script d'entraînement du modèle
├── .gitignore              # Fichiers et dossiers à ignorer par Git
├── README.md               # Ce fichier de documentation
├── cats_dogs_classifier.h5 # Modèle Deep Learning entraîné (NON versionné, voir Artefacts)
├── training_history.png    # Graphique de l'historique d'entraînement (NON versionné, voir Artefacts)
└── data/                   # Dataset (images de chats/chiens, NON versionné, voir Artefacts)
└── PetImages/
├── Cat/
└── Dog/

---

## ✨ Artefacts Générés (Non Versionnés sur Git)

Pour des raisons de taille de dépôt et de bonnes pratiques, certains fichiers générés ne sont pas versionnés sur Git, mais sont essentiels au fonctionnement de l'application :

* `cats_dogs_classifier.h5` : Le modèle de Deep Learning entraîné. Il est généré par `train_model.py` et copié dans l'image Docker par le `Dockerfile`.
* `training_history.png` : Un graphique illustrant les performances du modèle pendant l'entraînement, généré par `train_model.py`.
* `data/` : Le dossier contenant les images du dataset. Ce dataset est très volumineux et doit être téléchargé séparément (par exemple, depuis Kaggle). Le `Dockerfile` ne le copie pas car il n'est pas nécessaire à l'exécution de l'API après l'entraînement.

---

## 📋 Prérequis

Pour exécuter et reproduire ce projet, vous aurez besoin de :

* **Python 3.11+** : Pour le développement local et la création de l'environnement virtuel.
* **pip** : Le gestionnaire de paquets Python.
* **Docker Desktop** : Inclut Docker Engine et Docker Compose, nécessaire pour la conteneurisation et le déploiement local.
    * [Télécharger Docker Desktop](https://www.docker.com/products/docker-desktop/)
* **Un compte Kaggle** : Pour télécharger le dataset.
* **Un compte AWS** : Pour le déploiement sur EC2.
* **Un compte Docker Hub** : Pour publier votre image Docker.

---

## 🛠️ Guide de Démarrage Rapide

Suivez ces étapes pour configurer et lancer l'application.

### 1. Configuration Initiale du Projet

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/brandonvellien/deep-learning-mlops.git](https://github.com/brandonvellien/deep-learning-mlops.git)
    cd deep-learning-mlops
    ```
    

2.  **Créer et activer l'environnement virtuel Python :**
    ```bash
    python3 -m venv venv_dlops
    source venv_dlops/bin/activate # Ou `venv_dlops\Scripts\activate` sur Windows
    ```

3.  **Installer les dépendances Python :**
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn fastapi uvicorn[standard] python-multipart
    # Ou générez requirements.txt d'abord: pip freeze > requirements.txt puis pip install -r requirements.txt
    ```

4.  **Télécharger le Dataset :**
    * Allez sur la page Kaggle du "Cats and Dogs Classification Dataset".
    * Téléchargez le dataset.
    * Décompressez le contenu du dataset dans le dossier `data/` à la racine de votre projet. Assurez-vous que la structure est `data/PetImages/Cat/` et `data/PetImages/Dog/`.

### 2. Entraînement du Modèle

1.  **Lancer l'entraînement :**
    Exécutez le script d'entraînement du modèle. Il va télécharger MobileNetV2, entraîner le modèle et le sauvegarder sous `cats_dogs_classifier.h5`.
    ```bash
    python train_model.py
    ```
    *(Ajustez le chemin `path_to_dataset` dans `train_model.py` si votre dataset n'est pas directement sous `data/`.)*

2.  **Vérifier les artefacts :**
    Après l'entraînement, les fichiers `cats_dogs_classifier.h5` et `training_history.png` devraient apparaître à la racine de votre projet.

### 3. Conteneurisation de l'API

1.  **Générer le fichier `requirements.txt` :**
    * Assurez-vous que votre environnement virtuel est activé.
    ```bash
    pip freeze > requirements.txt
    ```

2.  **Construire l'image Docker :**
    * Assurez-vous d'être à la racine du projet (`deep-learning-mlops`).
    ```bash
    docker build -t cats-dogs-api:latest .
    ```

3.  **Tester l'image localement :**
    Lancez un conteneur et accédez à l'API.
    ```bash
    docker run -d --name cats-dogs-predictor-local -p 8000:8000 cats-dogs-api:latest
    ```
    Accédez à l'API via votre navigateur : `http://localhost:8000/docs`. Vous pouvez y tester l'endpoint `/predict/`.

### 4. Partage de l'Image sur Docker Hub

1.  **Se connecter à Docker Hub :**
    ```bash
    docker login
    ```

2.  **Tagger l'image :**
    ```bash
    docker tag cats-dogs-api:latest votre_nom_utilisateur_docker/cats-dogs-api:latest
    ```
    *(Remplacez `votre_nom_utilisateur_docker` par votre nom d'utilisateur Docker Hub.)*

3.  **Pousser l'image :**
    ```bash
    docker push votre_nom_utilisateur_docker/cats-dogs-api:latest
    ```
    *(Cette étape peut prendre un certain temps, car elle inclut le modèle de Deep Learning.)*

### 5. Déploiement sur AWS EC2

Cette section suppose que vous avez déjà une instance EC2 Ubuntu configurée avec un groupe de sécurité autorisant le SSH (port 22) et le port de l'API (8000) depuis internet. Le volume racine de l'instance doit être d'au moins 30 GiB.

1.  **Se connecter à l'instance EC2 via SSH :**
    ```bash
    ssh -i /chemin/vers/votre/cle.pem ubuntu@VOTRE_ADRESSE_IP_EC2
    ```
    *(Remplacez par le chemin de votre clé et l'IP de votre instance EC2.)*

2.  **Installer Docker sur l'instance :**
    ```bash
    sudo apt update
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ubuntu # Ajoutez l'utilisateur au groupe docker
    exit # Déconnectez-vous
    ssh -i /chemin/vers/votre/cle.pem ubuntu@VOTRE_ADRESSE_IP_EC2 # Reconnectez-vous
    ```

3.  **Se connecter à Docker Hub depuis l'instance :**
    ```bash
    docker login
    ```

4.  **Tirer l'image Docker :**
    ```bash
    docker pull votre_nom_utilisateur_docker/cats-dogs-api:latest
    ```

5.  **Exécuter le conteneur :**
    ```bash
    docker run -d --name cats-dogs-predictor-ec2 -p 8000:8000 votre_nom_utilisateur_docker/cats-dogs-api:latest
    ```

### 6. Tester l'API Déployée

Ouvrez votre navigateur et accédez à :

`http://VOTRE_ADRESSE_IP_EC2:8000/docs`

Vous devriez voir l'interface Swagger UI de votre API. Testez l'endpoint `/predict/` en téléchargeant une image pour vérifier la prédiction.

---

## 🗑️ Nettoyage

Pour arrêter et supprimer le conteneur local :

```bash
docker stop cats-dogs-predictor-local
docker rm cats-dogs-predictor-local
