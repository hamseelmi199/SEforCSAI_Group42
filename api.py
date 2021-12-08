# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
import numpy as np
import bot
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



@app.after_request
@cross_origin()
def add_header(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods = ['GET', 'POST'])
@cross_origin()
def home():
    if(request.method == 'GET'):
        data = "hello world"
        return jsonify(data);
        
    
  

# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000 / home / 10
# this returns 100 (square of 10)
@app.route('/chat/<string:mes>', methods = ['GET','POST'])
@cross_origin()
def disp(mes):
	botmes = bot.chatbot_response(mes)
	return jsonify(botmes);
    
# driver function
if __name__ == '__main__':

    app.run(host="0.0.0.0",port=5000,debug = True)
