import pandas as pd
import cv2
import os
import numpy as np  # 빈 이미지 생성을 위해 추가

# 설정
# ---------------------------------------------------------
# 데이터셋의 가장 상위 루트 (run_000)
BASE_DIR = '/home/sechankim/ros2_ws/src/dataset/valet_parking/run_000'

# CSV 파일 경로 (run_000 폴더 내부에 있다고 가정)
CSV_PATH = os.path.join(BASE_DIR, 'total_actions.csv')

# ---------------------------------------------------------

# CSV 로드
if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}")
    exit()

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
delay = 100 # 기본 재생 속도 (ms)
paused = False # 일시정지 상태 플래그

# 이미지 리사이즈 함수 (4개를 합칠 때 크기가 다르면 오류가 날 수 있으므로 통일)
def resize_img(img, width=480, height=360):
    if img is None:
        # 이미지가 없을 경우 검은 화면 표시 (경로 에러 확인용 텍스트 추가)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(blank, "No Image", (int(width/2)-50, int(height/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return blank
    return cv2.resize(img, (width, height))

while idx < len(df):
    row = df.iloc[idx]
    
    # 에피소드 폴더명과 각 카메라 파일명 가져오기
    episode_dir = str(row['episode']) # episode 컬럼이 숫자일 수도 있으므로 문자열 변환
    
    # 4개 카메라 경로 생성
    file_front = str(row['front_cam']).lstrip('/')
    file_left  = str(row['left_cam']).lstrip('/')
    file_right = str(row['right_cam']).lstrip('/')
    file_rear  = str(row['rear_cam']).lstrip('/')

    path_front = os.path.join(BASE_DIR, episode_dir, file_front)
    path_left  = os.path.join(BASE_DIR, episode_dir, file_left)
    path_right = os.path.join(BASE_DIR, episode_dir, file_right)
    path_rear  = os.path.join(BASE_DIR, episode_dir, file_rear)

    # 이미지 로드
    img_front = cv2.imread(path_front)
    img_left  = cv2.imread(path_left)
    img_right = cv2.imread(path_right)
    img_rear  = cv2.imread(path_rear)

    # 모든 이미지를 동일한 크기로 리사이즈 (가로 480, 세로 360 예시 - 필요시 수정)
    W, H = 480, 360
    img_front = resize_img(img_front, W, H)
    img_left  = resize_img(img_left, W, H)
    img_right = resize_img(img_right, W, H)
    img_rear  = resize_img(img_rear, W, H)

    # 각 이미지에 라벨 표시
    cv2.putText(img_front, "FRONT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img_left,  "LEFT",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img_right, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img_rear,  "REAR",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 2x2 그리드로 합치기
    # [Front] [Rear ]
    # [Left ] [Right]
    top_row = cv2.hconcat([img_front, img_rear])
    bot_row = cv2.hconcat([img_left, img_right])
    final_frame = cv2.vconcat([top_row, bot_row])

    # 화면 상단에 전체 정보 표시 (상태, Index, 조향값)
    status_text = "PAUSED" if paused else "PLAYING"
    status_color = (0, 0, 255) if paused else (0, 255, 0)

    info_main = f"[{status_text}] IDX: {idx} | Ep: {episode_dir}"
    info_val  = f"Linear: {row['linear_x']:.2f} | Steer: {row['angular_z']:.2f}"

    # 전체 화면의 좌측 상단(Front 이미지 위)에 텍스트 오버레이
    cv2.putText(final_frame, info_main, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(final_frame, info_val,  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Multi-Camera Data Viewer", final_frame)

    # 키 입력 처리 (로직 동일)
    key = cv2.waitKey(0 if paused else delay) & 0xFF

    if key == ord(' '):
        paused = not paused
    elif key == ord('q'): 
        break
    elif key == ord('d'):
        idx = min(len(df) - 1, idx + 1)
    elif key == ord('a'): 
        idx = max(0, idx - 1)
    elif key == ord('w'): 
        delay = max(1, delay - 10)
    elif key == ord('s'): 
        delay += 10
    
    if not paused:
        idx += 1
        if idx >= len(df): 
            paused = True
            idx = len(df) - 1

cv2.destroyAllWindows()