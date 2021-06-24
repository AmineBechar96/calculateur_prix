from flask import Flask, request
import pandas as pd
import numpy as np
import pymongo
import flask
import os
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    client = pymongo.MongoClient(
        'mongodb+srv://amine:testtest@cluster0.qlxh6.mongodb.net/test2?retryWrites=true&w=majority')
    db = client.test2
    collection_sell = db['Sell2']
    sell = pd.DataFrame(list(collection_sell.find({}, {'_id': False})))
    sell.set_index('id', inplace=True)

    # In[3]:

    sell_predict = sell.loc[:,
                   ['model', 'brand', 'notes', 'type', 'proDate', 'energie', 'ch', 'transmission', 'kilometrage',
                    'litre', 'price']]

    # In[4]:

    var_model_highend = 0
    var_model_medium = 0
    var_energie_essence = 0
    var_energie_gpl = 0
    var_transmission_manuelle = 0
    var_transmission_semi = 0

    # In[5]:

    filename = os.path.join(app.root_path, 'static', 'finalized.sav')
    filename2 = os.path.join(app.root_path, 'static', 'finalized2.sav')
    filename3 = os.path.join(app.root_path, 'static', 'finalized3.sav')
    filename4 = os.path.join(app.root_path, 'static', 'finalized4.sav')
    linear = os.path.join(app.root_path, 'static', 'linear.sav')

    # In[6]:

    label_enc = pickle.load(open(filename, 'rb'))
    label_enc2 = pickle.load(open(filename2, 'rb'))
    label_enc3 = pickle.load(open(filename3, 'rb'))
    label_enc4 = pickle.load(open(filename4, 'rb'))

    # In[7]:

    model = request.json['model']
    brand = request.json['brand']
    notes = request.json['notes']
    types = request.json['types']
    proDate = request.json['proDate']
    energie = request.json['energie']
    transmission = request.json['transmission']
    ch = request.json['ch']
    litre = request.json['litre']
    kilometrage = request.json['kilometrage']

    # In[8]:

    brand_num = label_enc.transform([brand])[0]
    notes_num = label_enc2.transform([notes])[0]
    types_num = label_enc3.transform([types])[0]
    model_num = label_enc4.transform([model])[0]

    # In[9]:

    linear = pickle.load(open(linear, 'rb'))

    # In[10]:

    sell_predict.reset_index(inplace=True)
    sell_predict['price'] = sell_predict['price'].astype('int')
    temp = sell_predict.copy()
    table = temp.groupby(['model'])['price'].mean()
    temp = temp.merge(table.reset_index(), how='left', on='model')
    bins = [0, 100, 200, 400]
    cars_bin = ['Budget', 'Medium', 'Highend']
    sell_predict['carsrange'] = pd.cut(temp['price_y'], bins, right=False, labels=cars_bin)
    model_binaire = sell_predict[sell_predict.model == model].loc[:, 'carsrange'].iloc[1]

    # In[11]:

    if model_binaire == 'Highend':
        var_model_highend = 1
    if model_binaire == 'Medium':
        var_model_medium = 1

    # In[ ]:

    # In[15]:

    if transmission == 'Manuelle':
        var_transmission_manuelle = 1
    if transmission == 'Semi Automatique':
        var_transmission_semi = 1

    # In[ ]:

    # In[16]:

    if energie == 'Essence':
        var_energie_essence = 1
    if energie == 'GPL':
        var_energie_gpl = 1

    # In[17]:

    value = np.array([[model_num, brand_num, notes_num, types_num, proDate, ch, litre, kilometrage, var_model_medium,
                       var_model_highend,
                       var_transmission_manuelle, var_transmission_semi, var_energie_essence, var_energie_gpl]])
    arrays = pd.DataFrame(data=value, columns=['model', 'brand', 'notes', 'type', 'proDate', 'ch', 'litre',
                                               'kilometrage', 'Medium', 'Highend', 'Manuelle', 'Semi Automatique',
                                               'Essence', 'GPL'])

    # In[18]:

    value = linear.predict(arrays)

    # In[19]:
    val = str((np.e ** value)[0])
    return val


if __name__ == '__main__':
    app.run()
