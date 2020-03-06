import flask
from PIL import Image
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import tensorflow 
from train import CNN_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import io

app = flask.Flask(__name__)
api = Api(app)

model = None


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# load weights into new model

model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"prediction" : None}

    
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(256, 256))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            prediction = model.predict(image)

            # Output 'Negative' or 'Positive' along with the score
            if prediction == 0:
                pred = 'Bad'
            else:
                pred = 'Good'

            # indicate that the request was a success
            data["prediction"] = pred    # return the data dictionary as a JSON response
    return flask.jsonify(data)



if __name__ == '__main__':
    app.run(debug=True)

