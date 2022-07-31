from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow import Graph
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import json

from numpy import array
import numpy as np
from keras.applications.inception_resnet_v2 import  InceptionResNetV2 , preprocess_input


from keras_preprocessing.text import tokenizer_from_json
import json
with open('./models/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

IR2 = InceptionResNetV2(weights='imagenet',include_top=False)


model=load_model('./models/my_model5.h5')


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'START'
    # iterate over the whole length of the sequence
    for i in range(1000):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0][-10:]

        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)

        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)

        # convert probability to integer
        yhat = np.argmax(yhat)

        # map integer to word
        word = word_for_id(yhat, tokenizer)

        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # Print the prediction
        print(' '+word, end='')
        # stop if we predict the end of the sequence
        if word == '\n</html>':
            break
    return in_text

# Create your views here.
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def about(request):
    return render(request,'about.html')

def wireframe(request):
    print(request)
    print(request.POST.dict())
    fileobj=request.FILES['filePath']
    fs=FileSystemStorage()
    global filepathname
    filepathname=fs.save(fileobj.name,fileobj)
    filepathname=fs.url(filepathname)
    global testimage

    testimage='.'+filepathname
    context={'filepathname':filepathname}
    return render(request,'wireframe.html',context)
def viewcode(request):
    test_image = image.img_to_array(image.load_img(testimage, target_size=(299, 299)))
    test_image = np.array(test_image, dtype=float)
    test_image = preprocess_input(test_image)
    test_features = IR2.predict(np.array([test_image]))

    list1=generate_desc(model, tokenizer, np.array(test_features), 10)
    (filepathname)
    context={'l1':list1,'filepathname':filepathname}
    return render(request,'viewcode.html',context)
