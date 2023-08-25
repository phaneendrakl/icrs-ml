from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
from django.template import RequestContext

from users.forms import (UserLoginForm, UserRegistrationForm)
from users.models import UserRegistrationModel


# Create your views here.
def user_login_check(request):
    context = {'form': UserLoginForm()}
    if request.method == 'POST':
        print('=000=' * 40, request.POST)
        form = UserLoginForm(request.POST)
        print('VALID:', form.is_valid())
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            print('UserName:', username)
            print('Password:', password)
            try:
                user = UserRegistrationModel.objects.get(username=username, password=password)
                print("User Object", user.status)
                status = user.status
                if status == 'activated':
                    request.session['username'] = user.username
                    request.session['password'] = user.password
                    return render(request, 'users/user_home.html')
                else:
                    messages.success(request, 'Your A/C has not been activated by admin.')
                    return render(request, 'user_login.html', context)
            except Exception as e:
                print("Got Exception: ", e)
                pass
            messages.error(request, 'Invalid username or password.')
    context = {'form': UserLoginForm()}
    return render(request, 'user_login.html', context)


def user_register_action(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your request has been submitted, Admin will get back to you soon.')
            context = {"form": UserRegistrationForm()}
            return render(request, 'register.html', context)
        else:
            messages.error(request, 'Email Already Exists.')

    context = {
        'form': UserRegistrationForm()
    }
    return render(request, 'register.html', context)


def add_data(request):
    return render(request, 'users/add_data.html')


def crop_prediction(request):
    if request.method == 'POST':
        N = request.POST.get('nitrogen')
        P = request.POST.get('Phosphorous')
        K = request.POST.get('Pottasium')
        ph = request.POST.get('ph')
        rainfall = request.POST.get('Rainfall')


def fertilizer(request):
    import pandas as pd
    import os
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT, 'fertilizer.csv')
    df = pd.read_csv(path)
    df = df[['Crop', 'N', 'P', 'K', 'pH', 'soil_moisture']]
    df = df.head(96).to_html
    return render(request, 'users/fertilizer.html', {'dat': df})


def raw_district_wise_yield_data(request):
    import pandas as pd
    import os
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT, 'raw_districtwise_yield_data.csv')
    pf = pd.read_csv(path)
    pf = pf[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area', 'Production']]
    pf = pf.head(240).to_html
    return render(request, 'users/raw_district_wise_yield_data.html', {'da': pf})


# def ML_Result(request):
#     return render(request, 'users/ML_Result.html')


def fertilizer(request):
    import pandas as pd
    import os
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT, 'fertilizer.csv')
    df = pd.read_csv(path)
    df = df[['Crop', 'N', 'P', 'K', 'pH', 'soil_moisture']]
    df = df.head(96).to_html
    return render(request, 'users/fertilizer.html', {'dat': df})


def view_data(request):
    return render(request, 'users/view_data.html', {})


def crop_recommendation(request):
    import pandas as pd
    import os
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT, 'crop_recommendation.csv')
    df = pd.read_csv(path)
    df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'CEC', 'label']]
    df = df.head(2250).to_html
    return render(request, 'users/crop_recommendation.html', {'dat': df})


def crop_sustainability3(request):
    import pandas as pd
    import os
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT, 'crop_sustainability3.csv')
    pf = pd.read_csv(path)
    pf = pf[['N', 'Phosphorous', 'Potassium', 'Ph', 'Soil_Moisture', 'CEC', 'SEASON', 'Suitable_Label']]
    pf = pf.head(2500).to_html
    return render(request, 'users/crop_sustainability3.html', {'da': pf})


def user_classification(request):
    from .utility import process_classification
    dt_accuracy, dt_precission, dt_f1_score, dt_recall = process_classification.build_decisiontree_model()
    nb_accuracy, nb_precission, nb_f1_score, nb_recall = process_classification.build_naive_model()
    svm_accuracy, svm_precission, svm_f1_score, svm_recall = process_classification.build_svm_model()
    lr_accuracy, lr_precission, lr_f1_score, lr_recall = process_classification.build_lregression_model()
    rf_accuracy, rf_precission, rf_f1_score, rf_recall = process_classification.build_random_model()
    # xgb_accuracy, xgb_precission, xgb_f1_score, xgb_recall = process_classification.build_xgboost_model()
    mlp_accuracy, mlp_precission, mlp_f1_score, mlp_recall = process_classification.build_neuralnetwork_model()

    dt = {'dt_accuracy': dt_accuracy, 'dt_precission': dt_precission, 'dt_f1_score': dt_f1_score,
          'dt_recall': dt_recall}
    nb = {'nb_accuracy': nb_accuracy, 'nb_precission': nb_precission, 'nb_f1_score': nb_f1_score,
          'nb_recall': nb_recall}
    svm = {'svm_accuracy': svm_accuracy, 'svm_precission': svm_precission, 'svm_f1_score': svm_f1_score,
           'svm_recall': svm_recall}
    lr = {'lr_accuracy': lr_accuracy, 'lr_precission': lr_precission, 'lr_f1_score': lr_f1_score,
          'lr_recall': lr_recall}
    rf = {'rf_accuracy': rf_accuracy, 'rf_precission': rf_precission, 'rf_f1_score': rf_f1_score,
          'rf_recall': rf_recall}
    # xgb = {'xgb_accuracy': xgb_accuracy, 'xgb_precission': xgb_precission, 'xgb_f1_score': xgb_f1_score,
    #    'xgb_recall': xgb_recall}
    mlp = {'mlp_accuracy': mlp_accuracy, 'mlp_precission': mlp_precission, 'mlp_f1_score': mlp_f1_score,
           'mlp_recall': mlp_recall}
    return render(request, 'users/Ml_result.html', {'dt': dt, 'nb': nb, 'svm': svm, 'lr': lr, 'rf': rf, 'mlp': mlp})


# def user_neuralnetwork(request):
#     from .utility import process_dl
#     accuracy, precission, f1_score, recall = process_dl.build_neural_model()
#     mlp = {'accuracy': accuracy, 'precission': precission, 'f1_score': f1_score, 'recall': recall}
#     return render(request, 'users/Ml_result.html', {'mlp': mlp})


def plotting(request):
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from .utility.process_classification import plotting
    # plt.figure(figsize=[10, 6], dpi=100)
    # plt.title('Accuracy Comparision Chart')
    # plt.xlabel('Algorithms')
    # plt.ylabel('Accuracy')
    # pf = sns.barplot(x=plotting.acc, y=plotting().model, palette='dark')
    from .utility.process_classification import plotting
    ll = plotting()
    return render(request, 'users/plotting.html', {'ll': ll})


def predict_ML(request):
    if request.method == "POST":
        n = request.POST.get("nitrogen")
        p = request.POST.get("phosphorous")
        k = request.POST.get("pottasium")
        ph = request.POST.get("ph")
        r = request.POST.get("rainfall")
        import numpy as np
        import pandas as pd
        from .utility.process_classification import build_neuralnetwork_model2
        # path = 'C:\\Users\\Rajesh\\Desktop\\dataset\\crop_recommendation.csv'
        # df = pd.read_csv(path)
        data = np.array([[n, p, k, ph, r]])
        result = build_neuralnetwork_model2(data)
        return render(request, "users/add_data.html", {"result": result})
    else:
        msg = print("Your Field Not Suitable for any Crop,Try with Newer Fields")
        return render(request, "users/add_data.html", {'msg': msg})

# def input(request):
#     return HttpResponse("Hiii")
