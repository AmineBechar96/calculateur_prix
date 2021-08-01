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
import datetime
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    client = pymongo.MongoClient(
        'mongodb+srv://amine:testtest@cluster0.qlxh6.mongodb.net/test2?retryWrites=true&w=majority')
    db = client.test2
    collection_sell = db['Sell']
    sell = pd.DataFrame(list(collection_sell.find({}, {'_id': False})))
    sell.set_index('id', inplace=True)

    # In[3]:

    sell_predict = sell.loc[:,['model','brand','notes','type','proDate','energie','ch','transmission','kilometrage','litre','price','location_wilaya','tdi']]

    # In[4]:

    var_model_highend = 0
    var_model_medium = 0
    var_energie_essence = 0
    var_energie_gpl = 0
    var_transmission_manuelle = 0
    var_transmission_semi = 0

    # In[5]:

    filename = os.path.join(app.root_path, 'static', 'finalizedd.sav')
    filename2 = os.path.join(app.root_path, 'static', 'finalizedd2.sav')
    filename3 = os.path.join(app.root_path, 'static', 'finalizedd3.sav')
    filename4 = os.path.join(app.root_path, 'static', 'finalizedd4.sav')
    filename5 = os.path.join(app.root_path, 'static', 'finalizedd5.sav')
    filename6 = os.path.join(app.root_path, 'static', 'finalizedd6.sav')
    linear = os.path.join(app.root_path, 'static', 'linear4.sav')

    # In[6]:

    label_enc = pickle.load(open(filename, 'rb'))
    label_enc2 = pickle.load(open(filename2, 'rb'))
    label_enc3 = pickle.load(open(filename3, 'rb'))
    label_enc4 = pickle.load(open(filename4, 'rb'))
    label_enc5 = pickle.load(open(filename5, 'rb'))
    label_enc6 = pickle.load(open(filename6, 'rb'))

    # In[7]:

    model = request.json['model']
    brand = request.json['brand']
    notes = request.json['notes']
    types = request.json['types']
    proDate = request.json['proDate']
    energie = request.json['energie']
    location = request.json['location']
    tdi = request.json['tdi']
    transmission = request.json['transmission']
    ch = request.json['ch']
    litre = request.json['litre']
    kilometrage = request.json['kilometrage']

    sell_predict['id'] = sell_predict.index
    car_detaille = sell_predict[
        (sell_predict.brand == brand) & (sell_predict.notes == notes) & (sell_predict.proDate == proDate)].groupby(
        ['brand', 'notes', 'proDate']).describe()
    sell_predict.set_index('id', inplace=True)
    prix_max = car_detaille.loc[:, 'price'].iloc[0, 1]

    # In[8]:

    brand_num = label_enc.transform([brand])[0]
    notes_num = label_enc2.transform([notes])[0]
    types_num = label_enc3.transform([types])[0]
    model_num = label_enc4.transform([model])[0]
    location_num = label_enc5.transform([location])[0]
    tdi_num = label_enc6.transform([tdi])[0]


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
    value = np.array([[model_num, brand_num, notes_num, types_num, proDate, ch, litre, location_num, kilometrage,
                       tdi_num, var_model_medium, var_model_highend,
                       var_transmission_manuelle, var_transmission_semi, var_energie_essence, var_energie_gpl,
                       prix_max]])
    arrays = pd.DataFrame(data=value, columns=['model', 'brand', 'notes', 'type', 'proDate', 'ch', 'litre',
                                               'location', 'kilometrage', 'tdi', 'Medium', 'Highend', 'Manuelle',
                                               'Semi Automatique', 'Essence', 'GPL', 'prix_max'])


    value = linear.predict(arrays)

    # In[19]:
    val = str((np.e ** value)[0])
    return val

@app.route('/light', methods=['GET', 'POST'])
def hello_world2():
    client = pymongo.MongoClient(
        'mongodb+srv://amine:testtest@cluster0.qlxh6.mongodb.net/test2?retryWrites=true&w=majority')
    db = client.test2
    collection_sell = db['Sell']
    sell = pd.DataFrame(list(collection_sell.find({}, {'_id': False})))
    sell.set_index('id', inplace=True)

    # In[3]:

    sell_predict = sell.loc[:,['model','brand','notes','proDate','energie','transmission','kilometrage','price','location_wilaya']]
    # In[4]:

    var_model_highend = 0
    var_model_medium = 0
    var_energie_essence = 0
    var_energie_gpl = 0
    var_transmission_manuelle = 0
    var_transmission_semi = 0

    # In[5]:

    filename = os.path.join(app.root_path, 'static', 'finalizeaD.sav')
    filename2 = os.path.join(app.root_path, 'static', 'finalizeaD2.sav')
    filename3 = os.path.join(app.root_path, 'static', 'finalizeaD3.sav')
    filename4 = os.path.join(app.root_path, 'static', 'finalizeaD4.sav')
    filename5 = os.path.join(app.root_path, 'static', 'finalizeaD5.sav')
    linear = os.path.join(app.root_path, 'static', 'linear5.sav')

    # In[6]:

    label_enc = pickle.load(open(filename, 'rb'))
    label_enc2 = pickle.load(open(filename2, 'rb'))
    label_enc3 = pickle.load(open(filename3, 'rb'))
    label_enc4 = pickle.load(open(filename4, 'rb'))
    label_enc5 = pickle.load(open(filename5, 'rb'))


    # In[7]:

    model = request.json['model']
    brand = request.json['brand']
    notes = request.json['notes']
    proDate = request.json['proDate']
    energie = request.json['energie']
    location = request.json['location']
    transmission = request.json['transmission']
    kilometrage = request.json['kilometrage']

    sell_predict['id'] = sell_predict.index
    car_detaille = sell_predict[
        (sell_predict.brand == brand) & (sell_predict.notes == notes) & (sell_predict.proDate == proDate)].groupby(
        ['brand', 'notes', 'proDate']).describe()
    sell_predict.set_index('id', inplace=True)
    prix_max = car_detaille.loc[:, 'price'].iloc[0, 1]

    # In[8]:

    brand_num = label_enc.transform([brand])[0]
    notes_num = label_enc2.transform([notes])[0]
    model_num = label_enc4.transform([model])[0]
    location_num = label_enc5.transform([location])[0]



    # In[9]:

    linear = pickle.load(open(linear, 'rb'))

    # In[10]:

    sell_predict.reset_index(inplace=True)
    sell_predict['price'] = sell_predict['price'].astype('int')
    temp = sell_predict.copy()
    table = temp.groupby(['model'])['price'].mean()
    temp = temp.merge(table.reset_index(), how='left', on='model')
    bins = [0, 150, 400, 800]
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
    value1 = np.array([[model_num, brand_num, notes_num, proDate,location_num,  kilometrage,
                        var_model_medium, var_model_highend,
                       var_transmission_manuelle, var_transmission_semi, var_energie_essence, var_energie_gpl,
                       prix_max]])

    arrays = pd.DataFrame(data=value1, columns=['model', 'brand', 'notes', 'proDate',
                                                'location','kilometrage', 'Medium', 'Highend', 'Manuelle',
                                               'Semi Automatique', 'Essence', 'GPL', 'prix_max'])


    value = linear.predict(arrays)

    # In[19]:
    val = str((np.e ** value)[0])
    return val

@app.route('/accidenter', methods=['GET', 'POST'])
def hello_world3():
    annee = request.json['annee']
    prix_model = request.json['prix_model']
    prix_utilisateur = request.json['prix_utilisateur']


    now = datetime.datetime.now()
    val = percent(prix_utilisateur , prix_model)
    year = now.year - annee
    taux_final = val + year
    print(val,taux_final)


    if ((taux_final > -30) & (taux_final < 3)):
        return "Trés conseiller : Cette voiture est dans une Excellente Etat"
    elif ((taux_final > 2) & (taux_final < 16)):
        return "Conseiller : Cette voiture est dans un Bon Etat"
    elif ((taux_final > 15) & (taux_final < 25)):
        return "Déconseiller : Cette voiture est dans une Mauvaise Etat"
    elif ((taux_final > 24) & (taux_final < 50)):
        return "Trop déconseiller : Cette voiture est dans une trés Mauvaise Etat"
    else :
        return "Entrer une valeur valide"


def percent(a, b):
        result = int(((b - a) * 100) / a)

        return result
    
    
    
if __name__ == '__main__':
    app.run()
