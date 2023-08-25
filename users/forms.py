from django import forms
from .models import UserRegistrationModel


class UserRegistrationForm(forms.ModelForm):
    name = forms.CharField(required=True,
                           max_length=100,
                           widget=forms.TextInput(
                               attrs={
                                   'class': 'form-control',

                               }
                           ))
    username = forms.CharField(required=True,
                               max_length=100,
                               widget=forms.TextInput(
                                   attrs={
                                       'class': 'form-control'
                                   }
                               ))
    email = forms.EmailField(widget=forms.EmailInput(
        attrs={
            'class': 'form-control',

        }
    ))
    status = forms.CharField(
        initial='waiting',
        widget=forms.HiddenInput(
            attrs={
                'class': 'form-control'
            }
        ))

    password = forms.CharField(required=True,
                               widget=forms.PasswordInput(
                                   attrs={
                                       'pattern': '(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}',
                                       'title': 'Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters',
                                       'class': 'form-control',
                                   }
                               ))

    class Meta:
        model = UserRegistrationModel
        fields = '__all__'



class UserLoginForm(forms.Form):
    username = forms.CharField(required=True,
                               label_suffix='',
                               max_length=100,
                               widget=forms.TextInput(
                                   attrs={
                                       'class': 'form-control'
                                   }
                               ))

    password = forms.CharField(required=True,
                               label_suffix= "",
                               widget=forms.PasswordInput(
                                   attrs={
                                       'pattern': '(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}',
                                       'title': 'Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters',
                                       'class': 'form-control',
                                   }
                               ))
