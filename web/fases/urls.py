from django.urls import path
from . import views

urlpatterns = [
    path("", views.procesar_recepcion_0, name="procesar_recepcion_0"),
    path("procesar_recepcion_1/", views.procesar_recepcion_1, name="procesar_recepcion_1"),
]