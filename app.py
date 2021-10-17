from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_text as text
import requests
import json

app = Flask(__name__)
CORS(app)

saved_model_path = 'model'
reloaded_model = tf.saved_model.load(saved_model_path)

@app.route('/predict/', methods=['GET'])
def prediction():
   tweet = request.args.get("tweet", None)
   input_tweet = [str(tweet)]
   print(f"got name {tweet}")
   response = {}
   
   if not tweet:
      response["ERROR"] = "No tweet found."
   else:
      reloaded_results = tf.sigmoid(reloaded_model(tf.constant(input_tweet)))
      response['SCORE'] = float(f'{reloaded_results[0][0]:.6f}')

   return jsonify(response)

