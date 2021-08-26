#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from fastai.vision import *
from werkzeug.utils import secure_filename
import io
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import requests
from PIL import Image
import os
import base64
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

classes = ['abiu',
 'acai',
 'acerola',
 'ackee',
 'alligator apple',
 'ambarella',
 'apple',
 'apricot',
 'araza',
 'avocado',
 'bael',
 'banana',
 'barbadine',
 'barberry',
 'bayberry',
 'beach plum',
 'bearberry',
 'bell pepper',
 'betel nut',
 'bignay',
 'bilimbi',
 'bitter gourd',
 'black berry',
 'black cherry',
 'black currant',
 'black mullberry',
 'black sapote',
 'blueberry',
 'bolwarra',
 'bottle gourd',
 'brazil nut',
 'bread fruit',
 "buddha's hand",
 'buffaloberry',
 'burdekin plum',
 'burmese grape',
 'caimito',
 'camu camu',
 'canistel',
 'cantaloupe',
 'cape gooseberry',
 'carambola',
 'cardon',
 'cashew',
 'cedar bay cherry',
 'cempedak',
 'ceylon gooseberry',
 'che',
 'chenet',
 'cherimoya',
 'cherry',
 'chico',
 'chokeberry',
 'clementine',
 'cloudberry',
 'cluster fig',
 'cocoa bean',
 'coconut',
 'coffee',
 'common buckthorn',
 'corn kernel',
 'cornelian cherry',
 'crab apple',
 'cranberry',
 'crowberry',
 'cupuacu',
 'custard apple',
 'damson',
 'date',
 'desert fig',
 'desert lime',
 'dewberry',
 'dragonfruit',
 'durian',
 'eggplant',
 'elderberry',
 'elephant apple',
 'emblic',
 'entawak',
 'etrog',
 'feijoa',
 'fibrous satinash',
 'fig',
 'finger lime',
 'galia melon',
 'gandaria',
 'genipap',
 'goji',
 'gooseberry',
 'goumi',
 'grape',
 'grapefruit',
 'greengage',
 'grenadilla',
 'guanabana',
 'guarana',
 'guava',
 'guavaberry',
 'hackberry',
 'hard kiwi',
 'hawthorn',
 'hog plum',
 'honeyberry',
 'honeysuckle',
 'horned melon',
 'illawarra plum',
 'indian almond',
 'indian strawberry',
 'ita palm',
 'jaboticaba',
 'jackfruit',
 'jalapeno',
 'jamaica cherry',
 'jambul',
 'japanese raisin',
 'jasmine',
 'jatoba',
 'jocote',
 'jostaberry',
 'jujube',
 'juniper berry',
 'kaffir lime',
 'kahikatea',
 'kakadu plum',
 'keppel',
 'kiwi',
 'kumquat',
 'kundong',
 'kutjera',
 'lablab',
 'langsat',
 'lapsi',
 'lemon',
 'lemon aspen',
 'leucaena',
 'lillipilli',
 'lime',
 'lingonberry',
 'loganberry',
 'longan',
 'loquat',
 'lucuma',
 'lulo',
 'lychee',
 'mabolo',
 'macadamia',
 'malay apple',
 'mamey apple',
 'mandarine',
 'mango',
 'mangosteen',
 'manila tamarind',
 'marang',
 'mayhaw',
 'maypop',
 'medlar',
 'melinjo',
 'melon pear',
 'midyim',
 'miracle fruit',
 'mock strawberry',
 'monkfruit',
 'monstera deliciosa',
 'morinda',
 'mountain papaya',
 'mountain soursop',
 'mundu',
 'muskmelon',
 'myrtle',
 'nance',
 'nannyberry',
 'naranjilla',
 'native cherry',
 'native gooseberry',
 'nectarine',
 'neem',
 'nungu',
 'nutmeg',
 'oil palm',
 'old world sycomore',
 'olive',
 'orange',
 'oregon grape',
 'otaheite apple',
 'papaya',
 'passion fruit',
 'pawpaw',
 'pea',
 'peanut',
 'pear',
 'pequi',
 'persimmon',
 'pigeon plum',
 'pigface',
 'pili nut',
 'pineapple',
 'pineberry',
 'pitomba',
 'plumcot',
 'podocarpus',
 'pomegranate',
 'pomelo',
 'prikly pear',
 'pulasan',
 'pumpkin',
 'pupunha',
 'purple apple berry',
 'quandong',
 'quince',
 'rambutan',
 'rangpur',
 'raspberry',
 'red mulberry',
 'redcurrant',
 'riberry',
 'ridged gourd',
 'rimu',
 'rose hip',
 'rose myrtle',
 'rose-leaf bramble',
 'saguaro',
 'salak',
 'salal',
 'salmonberry',
 'sandpaper fig',
 'santol',
 'sapodilla',
 'saskatoon',
 'sea buckthorn',
 'sea grape',
 'snowberry',
 'soncoya',
 'strawberry',
 'strawberry guava',
 'sugar apple',
 'surinam cherry',
 'sycamore fig',
 'tamarillo',
 'tangelo',
 'tanjong',
 'taxus baccata',
 'tayberry',
 'texas persimmon',
 'thimbleberry',
 'tomato',
 'toyon',
 'ugli fruit',
 'vanilla',
 'velvet tamarind',
 'watermelon',
 'wax gourd',
 'white aspen',
 'white currant',
 'white mulberry',
 'white sapote',
 'wineberry',
 'wongi',
 'yali pear',
 'yellow plum',
 'yuzu',
 'zigzag vine',
 'zucchini']

curr_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, static_folder='static',
            template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'Images')


# In[3]:


def top_matches(preds, n):
    lst = preds[2].tolist()
    p_lst = []
    for i in range(0, n):
        val = max(lst)
        p_lst.append(val)
        lst.remove(val)


    lst = preds[2].tolist()
    result_data = []
    for i in range(0, n):
        val = max(lst)
        index = lst.index(val)
        percent = p_lst[i]/(sum(p_lst))
        percent = round(percent*100, 2)
        result_data.append(classes[index].capitalize() + ' - ' + str(percent) + '%')
        lst.remove(val)
    
    return result_data

def fruit_list():
    fruit_lst = []
    for i in range(0, 262):
        fruit_lst.append(classes[i].capitalize())

    return fruit_lst

# In[4]:


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/fruit')
def fruit():
    fruit_lst = fruit_list()
    return render_template('fruit.html', fruit_list = fruit_lst)

# In[5]:

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/main', methods=['post', 'get'])
def main():
    try:
        img = None
        n = 5
        if request.method == 'POST':
            img_url = request.form.get('url')
            n = int(request.form.get('matches'))
            if img_url != None and img_url != "": 
                img_resp = requests.get(img_url, stream=True)
                img = open_image(img_resp.raw)
                img_src = img_url
            else:
                img_file = request.files['file']
                filename = secure_filename(img_file.filename)
                img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                img = open_image(img_file)
                img_src= os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # img_src = (os.path.join(app.config['UPLOAD_FOLDER']) + '/' + img_file.filename)

                
        # else:
        #     img_url = request.args.get('img')
        #     img_resp = requests.get(img_url, stream=True)
        #     img = open_image(img_resp.raw)
        model = load_learner('Fruit', 'deploy_final.pth')
        preds = model.predict(img)
        data = top_matches(preds, n)
        return render_template('main.html', data=data, image_source=img_src)
    except:
        data = ['Error: The fruit image you selected could not be identified. Please try uploading in a different format or a different image altogether.']
        return render_template('main.html', data=data)



# In[ ]:


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


# In[ ]:




