
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import requests
import io

#https://drive.google.com/file/d/1q_tpfhwz-5TicB5L2VZJkIiLskglyVJf
# Define the class names
class_names1 = ['Aban','Abe_Dua','Adikrahene_Dua','Adinkrahene','Adwera','Adwo','Agyinduwura','Akoben','Akofena','Akokonan','Akoma','Akoma_Ntoso','Ananse_Ntontan','Ani_Bere','Asase_Ye_Duru','Aya','Bese_Saka','Biribi_Wo_Soro','Bi_Nnka_Bi','Boa_Me_Na_Me_Boa_Wo','Dame_Dame','Denkyem','Dono','Duafe','Dwannimmen','Eban','Epa','Ese_Ne_Tekrema','Fafanto','Fawohudie','Fihankra','Fofo','Funtumfunafu_Denkyem_Funafu','Gye_Nyame','Hwemudua','Hye_Won_Hye','Kae_Me','Kete_Pa','Kintinkantan','Kojo_Baiden','Kontire_Ne_Akwamu','Krado','Kramo_Bone','Kuntinkantan','Kwatakye_Atiko','Mako','Mate_Masie','Mframadan','Mmere_Dane','Mmomudwan','Mmusuyidee','Mpatapo','Mpuannum','Nea_Onnim_No_Sua_A_Ohu','Nea_Ope_Se_Nkrofoo_Ye_Ma_Wo_No-_Ye_Saa_Ara_Ma_won','Nea_Ope_Se_Obedi_Hene','Nkonsonkonson','Nkontim','Nkuma_Kese','Nkyimu','Nkyinkyim','Nnonnowa','Nsaa','Nserewa','Nsoromma','Nyame_Akruma','Nyame_Biribi_Wo_Soro','Nyame_Dua','Nyame_Nnwu_Na_Mawu','Nyame_Nti','Nyame_Ye_Ohene','Nyansapo','Nya_Abotere','Odo_Nyera_Fie_Kwan','Ohene','Ohene_Aniwa','Ohene_Tuo','Ohen_Adwae','Okodee_Mmowere','Okuafo_Pa','Onyakopon_Adom_Nti_Biribiara_Beye_Yie_African_Adinkra_Weddin','Onyakopon_Aniwa','Onyakopon_Ne_Yen_Ntena','Osidan','Osram','Osram_Ne_Nsoromma','Owo_Foro_Adobe','Owuo_Atwedee','Owuo_Kum_Nyame','Pa_Gya','Sankofa','Sepow','Sesa_Woruban','Sunsum','Tabon','Tamfo_Bebre','Tumi_Te_Se_Kosua','Tuo_Ne_Akofena','Wawa_Aba','Wo_Nsa_Da_Mu_A','Wuforo_Dua_Pa_A']



# Define the image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# Load the model from the file-like object
model = keras.models.load_model('./models/AdinkraNet_model1.h5')
# Load the model from the saved file


# Define the class names
class_names = class_names1

# Create a Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image']

    # Save the image to a temporary file
    image.save('temp.jpg')

    # Load the image with the target size
    img = keras.preprocessing.image.load_img('temp.jpg', target_size=IMG_SIZE)

    # Preprocess the image
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Predict the class of the image using the model
    prediction = model.predict(img_array)

    # Get the index of the highest probability
    index = np.argmax(prediction)

    # Get the class name corresponding to the index
    class_name = class_names[index]

    # Delete the temporary file
    os.remove('temp.jpg')

    # Render the predict.html template with the class name
    return render_template('predict.html', class_name=class_name)

# Run the app
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')