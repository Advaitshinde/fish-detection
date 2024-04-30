# # app.py

# from flask import Flask, render_template, request
# from keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# import numpy as np

# app = Flask(__name__)

# model = load_model('FishModelClassifier.h5', compile=False)
# class_name = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
#               'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp', 'Green Spotted Puffer',
#               'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish',
#               'Long-Snouted Pipefish', 'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish',
#               'Silver Barb', 'Silver Carp', 'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']


# def predict(path):
#     img = load_img(path, target_size=(224, 224, 3))
#     img = img_to_array(img)
#     img = img / 255
#     img = np.expand_dims(img, [0])
#     answer = model.predict(img)
#     x = list(np.argsort(answer[0])[::-1][:5])

#     predictions = []
#     for i in x:
#         predictions.append({"className": class_name[i], "predVal": float(answer[0][i]) * 100})

#     y_class = answer.argmax(axis=-1)
#     res = class_name[int(y_class)]
#     return res, predictions


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             file_path = "static/uploaded_image.jpg"
#             file.save(file_path)
#             result, predictions = predict(file_path)
#             return render_template('result.html', result=result, predictions=predictions)
#     # Add a return statement to handle cases where there is no valid file uploaded or other errors.
#     return "Error: No file uploaded or invalid request."


# if __name__ == '__main__':
#     app.run(debug=True)
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('FishModelClassifier.h5', compile=False)
class_name = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
              'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp', 'Green Spotted Puffer',
              'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish',
              'Long-Snouted Pipefish', 'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish',
              'Silver Barb', 'Silver Carp', 'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']

@st.cache(allow_output_mutation=True)
def load_and_predict(image):
    img = load_img(image, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    x = list(np.argsort(answer[0])[::-1][:5])

    predictions = []
    for i in x:
        predictions.append({"className": class_name[i], "predVal": float(answer[0][i]) * 100})

    y_class = answer.argmax(axis=-1)
    res = class_name[int(y_class)]
    return res, predictions

def main():
    st.title("Fish Species Classifier")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        result, predictions = load_and_predict(uploaded_file)
        st.write("Result:", result)
        st.write("Predictions:")
        for pred in predictions:
            st.write(f"- {pred['className']}: {pred['predVal']:.2f}%")

if __name__ == "__main__":
    main()
