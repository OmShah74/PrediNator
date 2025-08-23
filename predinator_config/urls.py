from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('akinator/', include('game_app.urls')),
    path('', lambda request: redirect('akinator/play/', permanent=False)),
]