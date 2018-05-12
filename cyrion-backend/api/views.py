from io import BytesIO

import base64
import logging
import os
import re
from datetime import datetime
from PIL import Image
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

logger = logging.getLogger(__name__)

UPLOAD_PARENT_DIR = os.path.join(settings.BASE_DIR, 'net', 'datasets', 'basic_data')


@api_view(['POST'])
def basic_upload(request):
    samples = request.data['samples']
    logger.debug("Receiving basic upload of %d samples" % len(samples))
    # dest = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    dest = "test_samples"
    upload_dir = os.path.join(UPLOAD_PARENT_DIR, dest)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    logger.debug("Samples destination: %s" % upload_dir)
    samples = [base64_to_png(samples[i], i, upload_dir) for i in range(len(samples))]
    return Response({'message': 'Helloww bro!'}, status=200)


def remove_html_tags(img_data):
    return re.search(r'base64,(.*)', img_data).group(1)


def base64_to_png(img_html, sid, upload_dir):
    # ImgData is a base64 png image, it is converted to jpg to avoid transparent background
    filename = os.path.join(upload_dir, 'sample%d.jpg' % sid)
    logger.debug("Converting base64 '%s..%s' to '%s'"
                 % (img_html[:20], img_html[-20:], filename))
    data = remove_html_tags(img_html)
    img = Image.open(BytesIO(base64.b64decode(data)))
    img = img.convert('L')
    img.save(filename, format="JPEG")
