from django.shortcuts import render, redirect
from .forms import AdminLoginForm
from django.contrib import messages
from users.models import UserRegistrationModel
from django.core.paginator import Paginator


# Create your views here.

def admin_home(request):
    return render(request, 'admins/admin_home.html')


def admin_login_check(request):
    print('REQ', request)
    context = {
        'form': AdminLoginForm()
    }
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        if username == 'admin' and password == 'admin':
            return render(request, 'admins/admin_home.html')
        else:
            messages.error(request, 'Incorrect username or password')
            return render(request, 'admin_login.html', context)

    return render(request, 'admin_login.html', context)


def users_list(request):
    users_list = UserRegistrationModel.objects.all()

    paginator = Paginator(users_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        # 'users': UserRegistrationModel.objects.all(),
        'users_list': page_obj,
    }
    return render(request, 'admins/users_list.html', context)


def activate_user(request, id):
    user = UserRegistrationModel.objects.get(id=id)
    # print("STATUS:", user.status)

    if user.status == 'waiting':
        user.status = 'activated'
        user.save()
    else:
        user.status = 'waiting'
        user.save()
    return redirect('admins_:admin_home')
