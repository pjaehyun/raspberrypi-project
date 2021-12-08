import cv2
import datetime
import requests

abs_path = '/home/jaehyun/workspace/test/media/'
data = {}
def picture():
    dt = datetime.datetime.now()
    cur_time = str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour) + str(dt.minute) + str(dt.second)
    cam = cv2.VideoCapture(0)
    ret, image = cam.read()
    dst = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    file_name = cur_time + ".jpg"
    cv2.imwrite(abs_path + file_name, dst)
    with open(abs_path+file_name, 'rb') as f:
        data = f.read()
    cam.release()
    cv2.destroyAllWindows()
    return data
img = picture()
files = {'image':img}
api_url = 'http://localhost:8000/api/pred/'
r = requests.post(api_url, files=files)
print(r.json()['prediction'])
