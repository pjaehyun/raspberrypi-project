from django.urls import path, include
from . import views

urlpatterns = [
    path("pred/", views.image_class),
]
