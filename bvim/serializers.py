from rest_framework import serializers
from .models import Image

# print("IN SERIALIZERS")
# print(Image)

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = '__all__'