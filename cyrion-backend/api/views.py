from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(['GET'])
def basic_upload(request):
    return Response({'message': 'Hellow bro!'}, status=200 )