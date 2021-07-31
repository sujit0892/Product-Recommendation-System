from flask import Flask
from flask import request
import product_recomendation
from flask import jsonify

app = Flask(__name__)

@app.route('/getPopularProduct/', methods=['GET'])
def getPopularProduct():
    return jsonify(product_recomendation.getPopularProduct())

@app.route('/getPopularProductByCategory/', methods=[ 'POST'])
def getPopularProductByCategory():
    cat_id = request.get_data().decode("utf-8")
    print(cat_id)
    return jsonify(product_recomendation.getPopularProductByCategory(cat_id))

@app.route('/recomendation/', methods=[ 'POST'])
def recomendation():
    user_id = request.get_data().decode("utf-8")

    return jsonify(product_recomendation.recomendation(user_id))

@app.route('/searchRecomendation/', methods=[ 'GET'])
def searchRecomendation():
    text = request.get_data().decode("utf-8")
    return jsonify(list(product_recomendation.show_recommendations(text)))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
