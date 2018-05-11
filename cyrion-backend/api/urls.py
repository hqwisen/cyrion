from django.urls import path

from api import views

urlpatterns = [
    path('basic/upload', views.basic_upload)
]
