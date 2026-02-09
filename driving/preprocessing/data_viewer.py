import pandas as pd
import cv2
import os

# 설정
CSV_PATH = '/home/sechankim/ros2_ws/src/dataset/valet_parking/run_000/episode_006/actions.csv'

IMG_DIR = '/home/sechankim/ros2_ws/src/dataset/valet_parking/run_000/episode_006' # image_path가 'images/xxx.jpg'이므로 상위 폴더 지정

# IMAGE_PATH = 'left_image'
# IMAGE_PATH = 'left_image'
# IMAGE_PATH = 'left_image'
IMAGE_PATH = 'right_image'

df = pd.read_csv(CSV_PATH)
print("--------------------------------------------------")
print("🎮 조작법:")

print("    [W]")
print("[A] [S] [D] ")
print("  [Space] \n")

print(" - [W]: 재생 속도 빨라짐")
print(" - [A]: 이전장")
print(" - [S]: 재생 속도 느려짐")
print(" - [D]: 다음장 (정지 상태에서 한 장씩 이동 가능)")
print(" - [Space]: 일시정지 / 다시 재생")
print(" - [Q]: 종료")
print("--------------------------------------------------")

idx = 0
delay = 30 # 기본 재생 속도 (ms)
paused = False # 일시정지 상태 플래그

while idx < len(df):
    row = df.iloc[idx]
    # img_path = os.path.join(IMG_DIR, row['image_path'])
    img_path = os.path.join(IMG_DIR, row[f'{IMAGE_PATH}'])
    img = cv2.imread(img_path)

    if img is None:
        idx += 1
        continue

    # 화면에 정보 표시 (상태, Index, 조향값)
    status_text = "PAUSED" if paused else "PLAYING"
    color = (0, 0, 255) if paused else (0, 255, 0) # 정지 시 빨간색, 재생 시 녹색
    
    info = f"[{status_text}] IDX: {idx}"
    info_val = f"Linear: {row['linear_x']:.2f} | Steer: {row['angular_z']:.2f}"

    cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, info_val, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    
    cv2.imshow("Data Viewer", img)
    
    # 일시정지 상태면 waitKey(0)으로 키 입력을 무한 대기, 아니면 정해진 delay만큼 대기
    key = cv2.waitKey(0 if paused else delay) & 0xFF
    
    if key == ord(' '): # 스페이스바: 일시정지 토글
        paused = not paused
    elif key == ord('q'): 
        break
    elif key == ord('d'): # 다음장
        idx = min(len(df) - 1, idx + 1)
    elif key == ord('a'): # 이전장
        idx = max(0, idx - 1)
    elif key == ord('w'): # 빨라짐
        delay = max(1, delay - 10)
    elif key == ord('s'): # 느려짐
        delay += 10
    
    # 정지 상태가 아닐 때만 인덱스를 자동으로 증가시킴
    if not paused:
        idx += 1
        if idx >= len(df): # 끝까지 가면 자동 정지
            paused = True
            idx = len(df) - 1

cv2.destroyAllWindows()