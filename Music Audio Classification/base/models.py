from django.db import models

class Song(models.Model):

    name = models.CharField(max_length=255)
    song_id = models.CharField(max_length=255)
    
    def __str__(self):
        return "[%s] %s" % (self.song_id, self.name)
    

class Class(models.Model):
    class_id = models.IntegerField(primary_key=True)
    class_name = models.CharField(max_length=64)
    
    def __str__(self):
        return "[%s] %s" % (self.class_id, self.class_name)
    

class Song2(models.Model):
    song_id = models.IntegerField(primary_key=True)
    class_id = models.ForeignKey(Class, on_delete=models.CASCADE)
    
    def __str__(self):
        return "[%s] %s" % (self.song_id, self.class_id)
    
    
