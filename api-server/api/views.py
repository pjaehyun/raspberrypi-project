from rest_framework.decorators import api_view
from static.python import imagepred as ip
from django.http import JsonResponse


@api_view(['POST'])
def image_class(request):

    bImage = request.body

    bImage = bImage[102:]

    predict = ip.prediction(bImage)

    return JsonResponse({
        'prediction': predict,
    })
