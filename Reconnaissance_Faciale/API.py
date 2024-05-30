import streamlit as st
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

embedder = FaceNet()
target_size = (160, 160)
detection = MTCNN()
encoder = LabelEncoder()

def Extraire_Visage(image):
    image = image
    x, y, w, h = detection.detect_faces(image)[0]['box']
    face = image[int(y):int(y+h), int(x):int(x+w)]
    face = cv2.resize(face, target_size)
    return face

def Embedding(face_img):
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    face_embedding = embedder.embeddings(face_img)
    return face_embedding[0]

faces_embeddings = np.load("faces_embeddings_4classes.npz")
X_train = faces_embeddings['arr_0']
Y_train = faces_embeddings['arr_1']
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train)

model = pickle.load(open("svm_model_160x160.pkl","rb"))
model.fit(X_train, Y_train) 

st.sidebar.title("Systeme de reconnaissance faciale")
pages = ["Objectif du projet", "Reconnaissance faciale", "Detect facial in real time"] 
page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:
    st.write("### Objectif du projet")
    st.write("L'objectif principal de ce projet est de crée un système capable de détecter et de reconnaitre les visages humains à partir de ces caractéristiques faciales.")
    st.write("Le système doit être capable de détecter une personne en comparant son visage avec une base de données d'images faciales préalablement enregistrées.")
    st.write("Le système doit également pouvoir reconnaitre si une personne donnée est celle qu'elle prétend être en comparant son visage avec une image ou une vidéo capturée en temps réel.")
    st.image("image.jpg")
    st.write("#### Reconnaissance Faciale")
    st.write("Dans cette partie on va donner au système une image qui contient un seul visage, il va détecter le visage et nous dire l'identité de la personne.")
    st.write("#### Detect facial in real time")
    st.write("Dans cette partie on va faire la reconaissance faciale en temps reel")

elif page == pages[1]:
    st.write("### Reconnaissance faciale")
    file = st.file_uploader('Téléchargez votre image', type=['png','jpg','jpeg'])
    if file is not None:
        image = plt.imread(file)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if st.button("run"):
            faces = detection.detect_faces(img)
            x, y, w, h = faces[0]["box"]
            face = Extraire_Visage(img)
            face_embedding = Embedding(face)
            face_embedding = [face_embedding]
            ypred = model.predict(face_embedding)
            prob = model.predict_proba(face_embedding)
            if round(prob.max(), 2) < 0.50:
                nom = "Inconnu"
            else:
                nom = encoder.inverse_transform(ypred)[0]

            pos = (x, y - 10)  # position du text
            # Obtenir les coordonnées et la taille du texte
            text_size, _ = cv2.getTextSize(nom, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)
            text_width, text_height = text_size
            # Calculer les coordonnées du rectangle pour encadrer le texte
            rectangle_left_top = pos[0], pos[1] - text_height
            rectangle_right_bottom = pos[0] + text_width, pos[1]

            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)
            img = cv2.rectangle(img, rectangle_left_top, rectangle_right_bottom, (0, 0, 0), -1)
            img = cv2.putText(img, nom, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)

elif page == pages[2]:
    st.write("### Detect facial in real time")
    
    haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    cap = cv2.VideoCapture(0)

    if st.button("run"):
        while cap.isOpened(): #True:
            _, frame = cap.read()
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
            for x,y,w,h in faces:
                img = rgb_img[y:y+h, x:x+w]
                img = cv2.resize(img, (160,160)) # 1x160x160x3
                img = np.expand_dims(img,axis=0)
                ypred = embedder.embeddings(img)
                face_name = model.predict(ypred)
                final_name = encoder.inverse_transform(face_name)[0]
        
                prob = model.predict_proba(ypred)

                if round(prob.max(),2) < 0.50:
                    nom = "Inconnu"
                else:
                    nom = final_name
        
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2) 
                cv2.putText(frame, str(nom), (x,y-10), cv2.FONT_HERSHEY_TRIPLEX,1, (0,100,255), 1)
                 
            cv2.imshow("Face Recognition:", frame)
            if cv2.waitKey(1) == ord('q'):
                break # Appuis sur le bouton "q" pour sortir de boucle (Fermer le webcam)
            
        cap.release() 
        cv2.destroyAllWindows()    