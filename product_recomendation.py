import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import sklearn
from sklearn.decomposition import TruncatedSVD
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
plt.style.use("ggplot")

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root1",
                               pw="password",
                               db="shopcart"))

def getPopularProduct():
    review = pd.read_sql("SELECT * from review",engine)
    popular_products = pd.DataFrame(review.groupby('product')['rating'].count())
    most_popular = popular_products.sort_values('rating', ascending=False)
    return list(most_popular.index[0:10])

def getPopularProductByCategory(cat_id):
    review = pd.read_sql("SELECT * from review where category="+str(cat_id),engine)
    popular_products = pd.DataFrame(review.groupby('product')['rating'].count())
    most_popular = popular_products.sort_values('rating', ascending=False)
    return list(most_popular.index[0:10])

def recomendation(user_id):
    review = pd.read_sql("SELECT * from review",engine)
    ratings_utility_matrix = review.pivot_table(values='rating', index='user_id', columns='product', fill_value=0)
    X = ratings_utility_matrix.T
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    product_names = list(X.index)
    order = pd.read_sql("SELECT product from orders where user_id='"+user_id+"'",engine)
    Recommend = []
    products = order['product'].values
    for product in products:
        product_ID = product_names.index(product)
        correlation_product_ID = correlation_matrix[product_ID]
        Recommend.extend(list(X.index[correlation_product_ID > 0.90]))
    for product in products:
        Recommend.remove(product)
    random.shuffle(Recommend)
    return Recommend[0:9]

product_descriptions = pd.read_csv('product.csv')
product_descriptions = product_descriptions[['asin','description']]
product_descriptions = product_descriptions.dropna()
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions["description"])
X=X1

true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

def show_recommendations(product):
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    text =[]
    for ind in order_centroids[prediction[0],:10]:
        text.append(terms[ind])
    if product not in text:
        text.append(product)
    products =[]
    for name in text:
        p = pd.read_sql("SELECT asin from product where name like'%"+name+"%' or description like '%"+name+"%'",engine)['asin'].values
        products.extend(p)
    return set(products)
