from django.urls import path

from . import views

app_name = 'base'

urlpatterns = [
    # api
    path('song', views.song, name='get_song'),
    path('songs', views.songs, name='get_songs'),
]