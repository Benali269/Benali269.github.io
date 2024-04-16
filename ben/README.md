
# Reconnaissance Faciale

L'objectif de ce projet est de cree une API capable de reconnaitre les visages humains a partir d'une image ou d'une video en temps reel.  
Pour realiser ce projet on va utilisee que le language Python sur Annaconda. 

## Preparer notre environnement de travail

1. Telecharger et installer Annaconda  
2. Cree un environnemt virtuel pour ce projet  
- Ouvrir annanconda apres l'installation
- Lancer "Powershell pompt"
- taper: 'conda create -n nom_VirEnv' puis clique sur "Enter"
- taper: 'Y' puis clique sur "Enter" 
- taper : 'conda activate nom_VirEnv' puis clique sur "Enter"
- ne fermer pas le "Powershell"
3. installation des packages dans l'environnement "nom_VirEnv"
Installer les packages citez ci dessous avec cette commande: 'conda install -c conda-forge nom_packages'

- tensorflow
- keras
- opencv-Python
- scikit-learn
- keras-facenet
- pickle
- mtcnn
- streamlit
- numpy
- matplotlib
- os-path 

4. Lancer jupyterLab  
retourne sur annaconda et connecte toi avec l'environnemt qu'on a cree et puis lancer  jupyterlab ou l'installer d'abord avant de le lancer.

5. Deplacer le dossier "Reconnaissance_Faciale" dans l'emplacement des fichiers de code Anaconda par defaut c'est: "C:\Users\<votre_nom_utilisateur>"  
Ouvre le dossier "Reconnaissance_Faciale" sur jupyterLab 

On trouve dans ce dossier:

- datasets : un dossier qui contient nos donnees d'images 
- image_test : un dossier qui contient des images pour le test et le deployement
- main.ipynb : fichier code de notre projet 
- En_temps_reel.ipynb : fichier code pour la Reconnaissance en temps reel
- API.py : fichier code pour le deployement 
- faces_embeddings_4classes.npz : ce fichier contient les features et les etiquette 
- svm_model_160x160.pkl : notre model on la enregistrer 
- haarcascade_frontalface_default.xml : pour la detection des visages  

## Utilisation des codes 

1. main.ipynb
Tout en haut cliquer sur 'Kernel' puis cliquer sur 'Restart Kernel and clear outputs of All cells...'.  
Apres clique sur 'Run' puis cliquer sur 'Run All cells'  

2. En_temps_reel.ipynb
Fait les meme etapes qu'avec le fichier 'main.ipynb' et pour fermer le webcam cliquer sue le bouton "q" du clavier.

3. API.py 
Pour l'API ouvre une nouvelle fenetre sur jupyterlab puis ouvre directement le terminal puis ecrire la caommande suivant pour lancer l'API: 'streamlit run API.py' et cliquer sur 'Enter' 

## Documentation

[Detect Face](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/)  
[Real-Time Face detection](https://www.eeweb.com/real-time-face-detection-and-recognition-with-svm-and-hog-features/)

## Authors

- [@Benali](https://github.com/Benali269)  