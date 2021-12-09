import cv2
import datetime
import requests

abs_path = './media/'
data = {}
def picture():
    # 현재 날짜 시간
    dt = datetime.datetime.now()
    cur_time = str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour) + str(dt.minute) + str(dt.second)
    
    # 첫번째 카메라를 VideoCapture의 객체로 얻어옴
    cam = cv2.VideoCapture(0)
    
    # 이미지 프레임 읽기
    ret, image = cam.read()
    
    # 결과파일 -> tensorflow에서 이미지를 width 224, height 224로 분석하기 때문에 224,224로 리사이징(사이즈를 줄이기 위해 INTER_AREA사용)
    dst = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    file_name = cur_time + ".jpg"
    
    # 읽어온 이미지 저장
    cv2.imwrite(abs_path + file_name, dst)
    
    # 서버로 이미지 사진을 보내기위해 바이트 형식으로 방금 저장된 이미지를 읽어옴
    with open(abs_path+file_name, 'rb') as f:
        data = f.read()
    
    # cam(VideoCapture)객체 종료
    cam.release()
    cv2.destroyAllWindows()
    
    # 바이트 단위로 저장된 이미지파일 반환
    return data

# API를 호출하기 위한 과정
img = picture()
files = {'image':img} # 파이썬 딕셔너리 형식으로 file설정
api_url = 'http://localhost:8000/api/pred/'
r = requests.post(api_url, files=files) #POST방식으로 파일전송
print(r.json()['prediction'])
