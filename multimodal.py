import cv2
import numpy as np
from keras.models import load_model
import queue
import threading
import sounddevice as sd
import soundfile as sf
import time
from pydub import AudioSegment

# 설정 값
THRESHOLD = 300  # 음성 감지 임계값

# 얼굴 감지를 위한 Haar Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

expression_labels = ['Angry', 'Happy', 'Sad', 'Neutral']

model = load_model('allexmodel.hdf5')

# 비디오 캡처 시작
video_capture = cv2.VideoCapture(0)

# 표정을 저장하기 위한 큐(queue) 초기화
expression_queue = []
queue_max_size = 6  # 초당 6번 값을 저장하도록 설정

q = queue.Queue()
recorder = False
recording = False

def complicated_record():
    global recording
    with sf.SoundFile("data/output.wav", mode='w', samplerate=20000, subtype='PCM_16', channels=1) as file:
        with sd.InputStream(samplerate=20000, dtype='int16', channels=1, callback=complicated_save):
            while True:
                chunk = q.get()
                vol = max(chunk)
                if vol < THRESHOLD and not recording:
                    print("-")

                if vol >= THRESHOLD and not recording:
                    recording = True
                    print('start recording')

                if recording:
                    file.write(chunk)
                    now = time.time()
                    if (now - start_time) >= 8:
                        print('stop recording')
                        break

def complicated_save(indata, frames, time, status):
    q.put(indata.copy())

def start_audio():
    global recorder
    global recording
    recording = False
    recorder = threading.Thread(target=complicated_record)
    recorder.start()

def stop_audio():
    global recorder
    global recording
    recording = False
    recorder.join()

start_audio()
start_time = time.time()

while True:
    # 비디오에서 프레임 읽기
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # 흑백으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 감지된 얼굴 영역 추출 및 크기 조정
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (96, 96))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # 모델을 사용하여 감정 분석
        output = model.predict(face_roi)[0]
        expression_index = np.argmax(output)
        expression_label = expression_labels[expression_index]

        # 표정 큐에 현재 표정 추가
        expression_queue.append(expression_label)

        # 큐의 크기를 초과하면 가장 오래된 표정 제거
        if len(expression_queue) > queue_max_size:
            expression_queue.pop(0)

        # 프레임에 감정 표시
        cv2.putText(frame, expression_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow('Expression Recognition', frame)

    # 가장 빈번하게 나타나는 표정 찾기
    most_frequent_expression = max(set(expression_queue), key=expression_queue.count)

    # 6번 이상 표정을 저장하면 해당 표정을 반환
    if expression_queue.count(most_frequent_expression) >= queue_max_size:
        print("Most frequent expression:", most_frequent_expression)
        expression_queue = []  # 표정 큐 초기화

    # ESC 키로 종료
    key = cv2.waitKey(25)
    if key == 27:
        break

# 비디오 캡처 해제 및 창 닫기
video_capture.release()
cv2.destroyAllWindows()

# 오디오 녹음 중지
stop_audio()

# # 음성 파일 불러오기
song = AudioSegment.from_wav("data/output.wav")

# 음성 파일 처리 및 저장
louder_song = song + 30
louder_song.export("data/output.wav", format='wav')
