from django.urls import path
from . import views
app_name = 'users_'

urlpatterns = [
    path('user_register_action/', views.user_register_action, name='user_register_action'),
    path('', views.user_login_check, name='user_login_check'),
    path('add_data/', views.add_data, name='add_data'),
    path('crop_prediction/', views.crop_prediction, name='crop_prediction'),
    path('fertilizer',views.fertilizer,name='fertilizer'),
    path('raw_district_wise_yield_data',views.raw_district_wise_yield_data,name='raw_district_wise_yield_data'),
    # path('ML_Result',views.ML_Result,name="ML_Result"),
    path('view_data',views.view_data,name="view_data"),
    path('crop_recommendation',views.crop_recommendation,name='crop_recommendation'),
    path('crop_sustainability3',views.crop_sustainability3,name="crop_sustainability3"),
    path('user_classification',views.user_classification,name='user_classification'),
    path('plotting',views.plotting,name='plotting'),
    path('predict_ML',views.predict_ML,name='predict_ML'),
    #path('input',views.input,name='input'),

]
