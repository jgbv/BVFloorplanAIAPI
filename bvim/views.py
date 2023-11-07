from django.shortcuts import render

from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import Image
from .serializers import ImageSerializer
from PIL import Image as PILImage
from . import ParseFloorplan as pf
import os 
import shutil
# from django.urls import reverse


# class ImageUpload(APIView):
#     parser_classes = (FileUploadParser,)

#     def post(self, request, *args, **kwargs):
#         file_serializer = ImageSerializer(data=request.data)

#         if file_serializer.is_valid():
#             file_serializer.save()
#             return Response(file_serializer.data, status=status.HTTP_201_CREATED)
#         else:
#             return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ImageUpload(APIView):
    parser_classes = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        print("====POST====")
        # url = reverse('upload')
        # print(f"post url: {url}")
        print(f"request: {request}")
        print(f"request.data: {request.data}\n{request.data['file'].name}")
        file_serializer = ImageSerializer(data=request.data)
        print(f"file_serializer: {file_serializer}")
        print(f"file_serializer.is_valid(): {file_serializer.is_valid()}")
        if file_serializer.is_valid():
            file_serializer.save()

            # Get the uploaded image instance
            uploaded_image = file_serializer.instance
            # print(f"========uploaded_image attr: {dir(uploaded_image)}")
            # # Get image name and dimensions
            image_name = uploaded_image.file.name
            with PILImage.open(uploaded_image.file) as img:
                # image_dimensions = img.size
                # imgX = img.size[0]
                # imgY = img.size[1]
                savepath = os.path.join("bvim", "images",f"{request.data['file'].name}")
                img.save(savepath)

            # response_data = {
            #     'imagePath': image_name,
            #     'imageHeight': imgY,
            #     'imageWidth': imgX,
            #     "predictions": []
            # }
            # print("!=--------------------over here")
            # response_data = pf.ParseFloorplan.parse(img=img, image_path=uploaded_image.file)
            response_data = pf.ParseFloorplan.parse(image_path=savepath)
            print(f"!-=-=-=-=-=-=--=-=-response_data: {response_data}")
            # print("!=--------------------now here")
            
            response = Response(response_data, status=status.HTTP_201_CREATED)
            # response['Content-Disposition'] = f'attachment; filename={response_data["imagePath"]}'
            # return Response(response_data, status=status.HTTP_201_CREATED)
            ImageUpload.clear_images_folder()
            return response
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def clear_images_folder():
        folder_path = 'media/images'
        print(f"!------{folder_path} exists?: {os.path.exists(folder_path)}")
        if os.path.exists(folder_path):
            # try:
            #     shutil.rmtree(folder_path)
            #     os.mkdir(folder_path)
            #     print(f"!--------removed {folder_path}--------")
                
            # except Exception as e:
            #     print(f"Error clearing images folder: {e}")
            for f in os.scandir(folder_path):
                if not f.is_dir():
                    imgpath=f.path
                    try:
                        os.remove(f.path)
                        print(f"removed {imgpath}")
                    except Exception as e:
                        print(f"Error removing {f.path}: {e}")
        else:
            print("Images folder does not exist.")
