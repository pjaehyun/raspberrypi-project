import serial
from sense_hat import SenseHat   
sense = SenseHat()

port = '/dev/ttyACM0'      # 시리얼통신에 사용할 포트
brate = 9600         # 통신속도 지정
cmd = 'temp'

seri = serial.Serial(port, baudrate = brate, timeout = None)   # serial 객체 생성
print(seri.name)

seri.write(cmd.encode())

a = 1
while a:
   content = seri.readline()         # 아두이노 출력값 읽어오기
   if(int(content.decode()) < 10):      # 거리값이 10cm보다 작으면
      print(content.decode())      # 물체와의 거리가 몇cm인지 출력하고
      sense.show_message("<<<car<<<")   # sense_hat에 왼쪽에서 차가오는것을 표시
   else:               # 거리값이 10cm보다 크면   
      print("차가 오지 않습니다.")      # 차가 오지 않음을 출력
      print(content.decode())      # 물체와의 거리값만 출력