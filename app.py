from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
import catboost
import pickle
model = pickle.load( open( "finalized_model.sav", "rb" ) )
app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route("/")
def hello():
  return "Hello World!"

@app.route("/get_prediction", methods=['POST'])
def create_task():
    if not request.json:
        abort(400)
    df = pd.DataFrame(request.json, index=[0])  
    return jsonify({'result': model.predict(df)[0]}), 201

if __name__ == "__main__":
  app.run()