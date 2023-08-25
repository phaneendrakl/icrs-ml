from django.shortcuts import render
from users.forms import (UserRegistrationForm, UserLoginForm)
from admins.forms import AdminLoginForm


# Create your views here.
def home(request):
    return render(request, 'index.html')


def register(request):
    context = {'form': UserRegistrationForm()}
    return render(request, 'register.html', context)


def user_login(request):
    context = {'form': UserLoginForm()}
    return render(request, 'user_login.html', context)


def admin_login(request):
    context = {'form': AdminLoginForm()}
    return render(request, 'admin_login.html', context)
