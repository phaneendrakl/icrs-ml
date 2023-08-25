from django.urls import path
from . import views

app_name = 'admins_'

urlpatterns = [
    path('', views.admin_login_check, name='admin_login_check'),
    path('users_list/', views.users_list, name='users_list'),
    path('admin_home/', views.admin_home, name='admin_home'),
    path('activate_user/<str:id>', views.activate_user, name='activate_user')

]
