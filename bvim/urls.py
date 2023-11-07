from django.urls import path
from .views import ImageUpload

urlpatterns = [
    path('upload/', ImageUpload.as_view(), name='image-upload'),
    # path('upload/', ImageUpload.as_view(), name='upload'),
]