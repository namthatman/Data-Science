from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Song, Song2, Class
from functools import lru_cache

import feature_engineer as fe
import numpy as np
import tensorflow as tf

import json
import base64


@lru_cache(maxsize=10)
def get_class_name(class_id_id):
    try:
        class_obj = Class.objects.get(class_id=class_id_id)
        class_name = class_obj.class_name
        return class_name      
    except Exception as err:
        return 'Unknown'       


@csrf_exempt
def song(request):
    if request.method == 'GET':
        song_obj = Song2()
        data = request.GET
        song_ids = data.get('id')
        song_ids = int(song_ids)
        if song_ids <= 999 and song_ids >= 0:
            try:
                song = Song2.objects.get(song_id=int(song_ids))
                class_id_id = song.class_id_id
                class_name = get_class_name(class_id_id)
                
                res = {
                    'song_id': song.song_id,
                    'class_id': song.class_id_id,
                    'class_name': class_name
                }
                return HttpResponse(json.dumps(res))
            except Exception as err:
                input_image = fe.get_input_image(int(song_ids))
                classifier = tf.keras.models.load_model('cnn_model')
                result = classifier.predict(input_image)
                result = np.argmax(result,axis=1)
                class_id = result[0]
                
                song = Song2.objects.create(song_id=int(song_ids), class_id_id=int(class_id))
                class_name = get_class_name(int(class_id))
                
                res = {
                    'song_id': int(song_ids),
                    'class_id': int(class_id),
                    'class_name': class_name
                }
                
                return HttpResponse(json.dumps(res))
        else:
            return HttpResponse("Wrong parameter")
    else:
        return HttpResponse("Wrong request method")
    

@csrf_exempt
def songs(request):
    if request.method == 'GET':
        request = request.GET
        if request:
            song_obj = Song()
            song_ids = data.get('song_id')
            if song_id:
                data = Song.objects.get(song_id=song_id)
                return HttpResponse(json.dumps(data))
            else:
                return HttpResponse("Wrong parameter")
    else:
        return HttpResponse("Wrong request method")