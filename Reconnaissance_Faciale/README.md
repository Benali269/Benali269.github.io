# Reconnaissance Faciale

L'objectif de ce projet est de crée une API capable de reconnaitre les visages humains à partir d'une image ou d'une vidéo en temps réel.  
Pour réaliser ce projet on va utilisée le language Python sur Anaconda. 

## Preparer notre environnement de travail

1. Telecharger et installer Annaconda  

2. Cree un environnemt virtuel pour ce projet 

- Ouvrir annanconda apres l'installation
- Cliquer sur l'onglet "Environments"
- En bas cliquer sur l'onglet "Create"
- Saisir un nom de l'environment virtuel example "test" puis cocher python et choisir la derniere version de python
- Puis cliquer sur "Create"
- Reviens sur l'onglet "Home" et a gauche du l'onglet "Channels" choisir le nom de l'environment 

3. installation des packages dans l'environnement "test"

- Installer JupyterLab sur Anaconda puis le lancer 
- ouvrir un notebook

Installer les packages citez ci dessous avec cette commande: '!pip install nom_packages' 

- tensorflow
- keras
- opencv-python
- scikit-learn
- keras-facenet
- mtcnn
- streamlit
- matplotlib 

5. Deplacer le dossier "Reconnaissance_Faciale" dans l'emplacement des fichiers de code Anaconda par defaut c'est: "C:\Users\<votre_nom_utilisateur>".

Ouvre le dossier "Reconnaissance_Faciale" sur jupyterLab 

On trouve dans ce dossier:

- datasets : un dossier qui contient nos donnees d'images 
- image_test : un dossier qui contient des images pour le test et le deployement
- main.ipynb : fichier code de notre projet 
- En_temps_reel.ipynb : fichier code pour la Reconnaissance en temps reel
- API.py : fichier code pour l'API 
- faces_embeddings_4classes.npz : ce fichier contient les features et les etiquette 
- svm_model_160x160.pkl : notre model on la enregistrer 
- haarcascade_frontalface_default.xml : pour la detection des visages  

## Utilisation des codes 

1. main.ipynb

Tout en haut cliquer sur l'onglet 'Kernel' puis cliquer sur 'Restart Kernel and clear outputs of All cells...'.  
Apres clique sur l'onglet 'Run' puis cliquer sur 'Run All cells'  

2. En_temps_reel.ipynb

Fait les memes etapes qu'avec le fichier 'main.ipynb' et pour fermer le webcam cliquer sur le bouton "q" du clavier.

3. API.py 

Pour l'API ouvre une nouvelle fenetre sur jupyterlab puis ouvre directement le terminal puis ecrire la caommande suivant pour lancer l'API: 'streamlit run API.py' et cliquer sur 'Enter' 

## Documentation

[Detect Face](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/)  
[Real-Time Face detection](https://www.eeweb.com/real-time-face-detection-and-recognition-with-svm-and-hog-features/) \
[Reconnaissance Faciale](https://www.kaspersky.fr/resource-center/definitions/what-is-facial-recognition)

## Authors

- [@Benali](https://github.com/Benali269)  