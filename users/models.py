from django.db import models


# Create your models here.
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    username = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    email = models.CharField(unique=True, max_length=100)

    status = models.CharField(max_length=100, default="waiting")

    def __str__(self):
        return str(self.username)

    class Meta:
        db_table = 'UserRegistrations'
