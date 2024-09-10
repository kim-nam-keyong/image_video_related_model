#!/usr/bin/env python
# coding: utf-8

# ### 데이터 전처리 

# pandas - 2.0.3
# numpy  - 1.23.5
# scikit-learn - 1.5.1
# imbalanced-learn - 0.12.3 
# pytorch -  2.4.0+cu118
# tensorflow - 2.10.0
# CUDA - 11.8
# cudnn - 8.7.0

# In[1]:


import pandas as pd
import numpy as np

print(pd.__version__) # 2.0.3
print(np.__version__) # 1.23.5


# In[2]:


import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# GPU가 사용 가능한지 확인
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 모든 GPU 메모리를 동적으로 할당하도록 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs {gpus} are available and memory growth is set.")
        
        # GPU 이름 출력
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"GPU Name: {details['device_name']}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs are available.")


# In[3]:


import torch

# PyTorch 버전 확인
print(f"PyTorch version: {torch.__version__}")

# CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    print(f"CUDA is available. PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")


# In[4]:


df_target = pd.read_csv("load_npy_img.csv",index_col=0)
df_target


# #### 이미지 데이터 전처리 efficientnet을 사용하여 특징벡터 추출 후 , DF에 따로 저장하여 load하기 편하게 사용 

# In[6]:


# 원본데이터로  , resize , nparray float32 , efficientnet preprocess 진행 

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import numpy as np
import pandas as pd
import os
import psutil
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time
import subprocess

# GPU 사용 여부 확인 및 메모리 성장 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 사용 가능 및 메모리 성장 설정 완료.")
    except RuntimeError as e:
        print(e)
else:
    print('GPU를 사용할 수 없습니다.')

class FeatureExtractor:
    def __init__(self, df, batch_size=32, model_name='efficientnetb0'):  # 배치 크기 늘리기
        self.df = df.dropna(subset=['img_file'])  # NaN 값 제거
        self.df['img_file'] = self.df['img_file'].astype(str)  # 문자열로 변환
        self.batch_size = batch_size
        self.model = self.load_model(model_name)
    
    def load_model(self, model_name):
        if model_name == 'efficientnetb0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        else:
            raise ValueError("지원되지 않는 모델 이름입니다.")
        return base_model

    def extract_features(self, img_array):
        features = self.model.predict(img_array, verbose=0)  # verbose=0으로 설정하여 진행 상황 메시지 비활성화
        return features

    def preprocess_image(self, img_path):
        try:
            # 이미지 로드
            img = Image.open(img_path)
            # 이미지 리사이즈
            img = img.resize((224, 224))
            # EfficientNet 전처리
            img_array = np.array(img, dtype=np.float32)
            img_array = preprocess_input(img_array)  # EfficientNetB0의 전처리 함수 사용
            return img_array
        except Exception as e:
            print(f"이미지 전처리 실패: {img_path}, 오류: {e}")
            return None

    def process_images(self, df_part):
        img_paths = df_part['img_file'].tolist()
        
        features_list = []
        with ThreadPoolExecutor() as executor:
            img_arrays = list(executor.map(self.preprocess_image, img_paths))
        
        img_arrays = [img for img in img_arrays if img is not None]
        if not img_arrays:
            return features_list
        
        img_arrays = np.array(img_arrays, dtype=np.float32)
        for i in range(0, len(img_arrays), self.batch_size):
            batch_arrays = img_arrays[i:i + self.batch_size]
            features = self.extract_features(batch_arrays)
            features_list.extend(features)
        
        return features_list

# 메모리 사용량 모니터링 함수
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"메모리 사용량: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# GPU 메모리 사용량 모니터링 함수 (nvidia-smi 사용)
def print_gpu_memory_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            print(f"GPU 메모리 사용량: {used} MiB / {total} MiB")
        else:
            print(f"nvidia-smi 오류: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi 명령어를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있는지 확인하세요.")

# 특징 추출 실행
df_sample = df_target  # 예제 데이터프레임 로드
extractor = FeatureExtractor(df_sample, batch_size=32)  # 배치 크기 늘리기

# 데이터프레임을 10등분하여 처리 (더 작은 단위로 나누기)
num_parts = 10
df_parts = np.array_split(df_sample, num_parts)

# 임시 파일 저장 경로
temp_dir = 'temp_features'
os.makedirs(temp_dir, exist_ok=True)

# tqdm을 사용하여 파트별 진행 상황 표시
part_times = []
for idx, df_part in enumerate(tqdm(df_parts, desc="Processing parts")):
    start_time = time.time()  # 파트 시작 시간 기록
    print(f"파트 {idx + 1}/{num_parts} 처리 중")  # 디버깅
    features = extractor.process_images(df_part)
    if features:  # 빈 리스트가 아닌 경우에만 저장
        features_df_part = pd.DataFrame(features)
        temp_file_path = os.path.join(temp_dir, f'features_part_{idx}.csv')
        features_df_part.to_csv(temp_file_path, index=False)
    end_time = time.time()  # 파트 종료 시간 기록
    elapsed_time = end_time - start_time  # 경과 시간 계산
    part_times.append(elapsed_time)
    
    # 평균 처리 시간 계산
    avg_time_per_part = np.mean(part_times)
    remaining_parts = num_parts - (idx + 1)
    estimated_remaining_time = avg_time_per_part * remaining_parts
    
    tqdm.write(f"파트 {idx + 1}/{num_parts} 처리 완료, 경과 시간: {elapsed_time:.2f} 초")
    tqdm.write(f"예상 남은 시간: {estimated_remaining_time:.2f} 초")
    
    print_memory_usage()  # 메모리 사용량 출력
    print_gpu_memory_usage()  # GPU 메모리 사용량 출력

# 임시 파일에서 데이터 로드 및 결합
feature_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
if feature_files:
    raw_features_df = pd.concat([pd.read_csv(f) for f in feature_files], ignore_index=True)
else:
    raw_features_df = pd.DataFrame()

# 임시 파일 삭제
for f in feature_files:
    os.remove(f)
os.rmdir(temp_dir)

# 특징 데이터프레임 확인
raw_features_df

raw_features_df.to_csv("raw_feature.csv",index=False)


# #### 비디오 데이터 전처리 , 매핑 실시 할 예정 09 07 01 53 기준 

# In[23]:


# 모든 비디오 파일 데이터 경로 수집 
import os 
import pandas as pd

base_path = './01.데이터/1.Training/원천데이터/TS2'

file_path = []

for root, dirs, dir in os.walk(base_path):
    if 'video' in dirs:
        video_path = os.path.join(root, 'video')
        for file in os.listdir(video_path):
            file_path.append(os.path.join(video_path, file))

df = pd.DataFrame(file_path, columns=['video_file'])
df

#df.isnull().sum() #0


# In[41]:


import os
import pandas as pd
import cv2

frame_counts = []

for file_path in df['video_file']:
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(frame_count)
    
    else: 
        frame_counts.append(None)
    cap.release()

df['frame_count'] = frame_counts
df


# In[72]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
plt.xlabel('Frame Count', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distribution of Frame Counts', fontsize=16)

sns.histplot(df_video['frame_count'], bins=20, kde=True, color='skyblue', edgecolor='black')

plt.xlim(left=-50)

#sns.despine(left=False, bottom=True)

plt.show()


# In[5]:


import pandas as pd
df_video = pd.read_csv("merged_video.csv")
df_video['occupant_id'].value_counts()  # occupant1? 1명의 데이터만 존재함. 

df_video


# In[117]:


import os
import glob
import json
import pandas as pd

# df_video DataFrame 생성
df_video = pd.read_csv("./df_video.csv")

# df_video의 내용을 확인
print("df_video 내용 확인:")
print(df_video.head())

json_dir = './01.데이터/1.Training/라벨링데이터/TL2'
json_files = glob.glob(os.path.join(json_dir, '**/*.json'), recursive=True)
results = []

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        video_id = data['metadata']['video_id']
        scene_id = data['scene_info']['scene_id']
        category_name = data['scene_info']['category_name']
        occupant_ids = [occupant['occupant_id'] for occupant in data['occupant_info']]

        # video_id와 scene_id를 df_video의 video_file 경로 형식에 맞게 변환
        video_file_pattern = f"\\{video_id}\\{scene_id}\\video\\{scene_id}.mp4"
        
        # 패턴과 일치하는지 확인
        matching_files = df_video[df_video['video_file'].str.contains(video_file_pattern, regex=False)]
        if not matching_files.empty:
            print(f"Matching files found for video_id {video_id} and scene_id {scene_id}:")
            print(matching_files)
            for occupant_id in occupant_ids:
                results.append({
                    'video_id': video_id,
                    'scene_id': scene_id,
                    'category_name': category_name,
                    'occupant_id': occupant_id,
                    'video_file': matching_files.iloc[0]['video_file']  # df_video의 video_file 값을 사용
                })
        else:
            print(f"No matching files found for video_id {video_id} and scene_id {scene_id} with pattern {video_file_pattern}")

result_df = pd.DataFrame(results)
print("Result DataFrame:")
print(result_df)


# In[124]:


#pd.concat([df_video, result_df],axis=1)  #53685

#result_df['video_file'] = result_df['video_file'].apply(lambda x: './01.데이터/1.Training/원천데이터/TS2' + x)


merged_df = pd.merge(df_video, result_df, on='video_file', how='left')
merged_df.to_csv("merged_video.csv",index=False)


# In[ ]:


# 프레임을 152로 고정하고 , 큰 프레임은 보간법을 이용하여 작은 프레임으로 변환 , 작은 프레임은 중복하여 사용한다. 
# 53000개 정도의 비디오 데이터를 모두 돌리면 120 시간이 걸려서 일단 5000개 정도로 줄여서 5000개의 비디오 파일의 프레임 152로 고정해서 특징벡터를 따왔다. 
# 특징벡터로 하려면 시간이 너무 오래걸리는데 일단 보류 . 
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
from tqdm import tqdm
import psutil
import subprocess
import time

#출력로그 제한.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU 사용 여부 확인 및 메모리 성장 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 사용 가능 및 메모리 성장 설정 완료.")
    except RuntimeError as e:
        print(e)
else:
    print('GPU를 사용할 수 없습니다.')

# 영상에서 프레임 추출한 데이터 전처리
def extract_features_from_frame(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(frame_rgb, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

# 영상에 각기 다른 프레임을 추출한 데이터를 보간하여 통일된 프레임 수로 변환
def interpolate_frames(frames, target_frames):
    current_frames = len(frames)
    if current_frames >= target_frames:
        step = current_frames / target_frames
        interpolated_frames = []
        for i in range(target_frames):
            idx = int(i * step)
            if idx < current_frames - 1:
                alpha = (i * step) - idx
                interpolated_frame = (1 - alpha) * frames[idx] + alpha * frames[idx + 1]
                interpolated_frames.append(interpolated_frame)
            else:
                interpolated_frames.append(frames[idx])
        return interpolated_frames
    
    interpolated_frames = frames.copy()
    while len(interpolated_frames) < target_frames:
        interpolated_frames.append(frames[len(interpolated_frames) % current_frames])
    
    return interpolated_frames[:target_frames]

# 비디오 파일 전처리 및 프레임별 특징추출 함수.
def preprocess_video(video_path, model, target_frames=152):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            features = extract_features_from_frame(frame, model)
            frames.append(features)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    cap.release()
    
    frames = interpolate_frames(frames, target_frames)
    
    return np.array(frames)

#
def load_videos_and_extract_features(df, model, target_frames=152):
    features_list = []
    total_videos = len(df)
    
    for video_path in tqdm(df['video_file'], desc="Processing videos", unit="video"):
        video_features = preprocess_video(video_path, model, target_frames)
        if video_features is not None:
            features_list.append(video_features)
        else:
            features_list.append(None)
    
    df['features'] = features_list

# EfficientNetB0 모델 로드
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# 비디오 파일에서 특징 추출
load_videos_and_extract_features(df_video_copy, efficientnet_model)

# 결과 출력
print(df_video_copy)


# ### 이미지 데이터기반 모델링 

# #### 기본 모델 형식 파라미터튜닝, 파인튜닝 X

# In[28]:


import numpy as np      #이미지 데이터 np.array 형태로 변환
import time
train_final = pd.read_csv("./train_final.csv")

start_time = time.time()
X_img = np.array(train_final['features'].apply(eval).tolist())
print("Data preprocessing time: {:.2f} seconds".format(time.time() - start_time))


# In[1]:


#np.savetxt('X_img.csv', X_img, delimiter=',')
import pandas as pd
train_final = pd.read_csv("./train_final.csv")


# In[2]:


import numpy as np 
X_img = np.loadtxt('./X_img.csv', delimiter=',')
#X_img = np.load('./X_img.npy')
X_img[1].shape


# In[3]:


y_img = train_final['label']
print(len(y_img))


# In[16]:


X_img_train , X_img_val, y_img_train, y_img_val = train_test_split(X_img, y_img , test_size = 0.2 ,  random_state =42)


# #### 시각화 모듈 

# In[20]:


# 전처리된 이미지 파일의 특징 벡터들의 분포를 확인 ,,, 이 파일은 이미지 데이터들을 .npy로 저장한 파일임 그리고 그 npy로 변환된 파일들의 특징벡터를 따온 것
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import time

sample_size = X_img.shape[0]

# 샘플링 수행
np.random.seed(42)  # 재현성을 위해 시드 설정
indices = np.random.choice(X_img.shape[0], sample_size, replace=False)
X_sample = X_img[indices]
y_sample = y_img[indices]

# PCA 소요 시간 측정
start_time = time.time()
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_sample)
end_time = time.time()
print(f"PCA 소요 시간: {end_time - start_time:.2f} 초")


def plot_3d(X, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_title(title)
    plt.show()

# PCA 결과 시각화
plot_3d(X_pca, y_sample, "PCA 3D Visualization")


# In[14]:


## 이건 원본데이터를 resize 랑 , preprocesse_input  efficientnet에 맞춰서 진행한 데이터 분포인데 생각보다 균형이 잡혀보임. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import time

sample_size = X_img.shape[0]

# 샘플링 수행
np.random.seed(42)  # 재현성을 위해 시드 설정
indices = np.random.choice(X_img.shape[0], sample_size, replace=False)
X_sample = X_img[indices]
y_sample = y_img[indices]

# PCA 소요 시간 측정
start_time = time.time()
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_sample)
end_time = time.time()
print(f"PCA 소요 시간: {end_time - start_time:.2f} 초")


def plot_3d(X, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_title(title)
    plt.show()

# PCA 결과 시각화
plot_3d(X_pca, y_sample, "PCA 3D Visualization")


# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time


sample_size = X_img.shape[0]

# 샘플링 수행
np.random.seed(42)  # 재현성을 위해 시드 설정
indices = np.random.choice(X_img.shape[0], sample_size, replace=False)
X_sample = X_img[indices]
y_sample = y_img[indices]

# PCA 소요 시간 측정
start_time = time.time()
pca = PCA(n_components=2)  # 2차원으로 변경
X_pca = pca.fit_transform(X_sample)
end_time = time.time()
print(f"PCA 소요 시간: {end_time - start_time:.2f} 초")

def plot_2d(X, y, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# PCA 결과 시각화
plot_2d(X_pca, y_sample, "PCA 2D Visualization")


# In[15]:


# 생각보다 전처리했을때 사진 데이터보다 , 역시 원본데이터를 특징추출한 데이터가 조금 더 균형이 잡아 보임

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time


sample_size = X_img.shape[0]

# 샘플링 수행
np.random.seed(42)  # 재현성을 위해 시드 설정
indices = np.random.choice(X_img.shape[0], sample_size, replace=False)
X_sample = X_img[indices]
y_sample = y_img[indices]

# PCA 소요 시간 측정
start_time = time.time()
pca = PCA(n_components=2)  # 2차원으로 변경
X_pca = pca.fit_transform(X_sample)
end_time = time.time()
print(f"PCA 소요 시간: {end_time - start_time:.2f} 초")

def plot_2d(X, y, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# PCA 결과 시각화
plot_2d(X_pca, y_sample, "PCA 2D Visualization")


# In[106]:


import matplotlib.pyplot as plt

# 그래프 크기 설정
plt.figure(figsize=(10, 6))

# 바 그래프 생성
ax = train['label'].value_counts().plot(kind='bar', color='skyblue')

# 각 바의 중앙에 숫자 표시
for p in ax.patches:
    ax.annotate(str(p.get_height()), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

# 그래프 제목과 축 레이블 추가 (선택 사항)
ax.set_title('Label Distribution')
ax.set_xlabel('Labels')
ax.set_ylabel('Counts')

# 그래프 표시
plt.show()


# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time

sample_size = int(0.1*X_img.shape[0])
np.random.seed(42)  # 재현성을 위해 시드 설정
indices = np.random.choice(X_img.shape[0], sample_size, replace=False)
X_sample = X_img[indices]
y_sample = y_img[indices]

# t-SNE 소요 시간 측정
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_sample)
end_time = time.time()
print(f"t-SNE 소요 시간: {end_time - start_time:.2f} 초")

def plot_2d(X, y, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# t-SNE 결과 시각화
plot_2d(X_tsne, y_sample, "t-SNE 2D Visualization")

## 특징벡터로 tsne 비선형 차원축소로 라벨2개이기에 , 1280 길이인 특징벡터를 길이가 2인 벡터로 줄여서 라벨의 분포를 확인해도 제대로 군집이 되어있지 않음. 
## 그래서 아래와 같은 accuracy 자체가 70% 정도로 낮은 값을 가지는 것으로 확인된다. 


# In[18]:


# 원본데이터의 1280길이를 가진 특징벡터를 2차원으로 축소 후 시각화를 했는데 사실 1280의 길이를 2로 줄이는 것이기에 데이터 손실이 굉장히 큼,
# 그냥 1차원적으로 clustering이 잘 되었는지 확인하는 정도의 용도
# 결론은 이것만 가지고는 정확한 판단이 불가함.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time

sample_size = int(0.5*X_img.shape[0])
np.random.seed(42)  # 재현성을 위해 시드 설정
indices = np.random.choice(X_img.shape[0], sample_size, replace=False)
X_sample = X_img[indices]
y_sample = y_img[indices]

# t-SNE 소요 시간 측정
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_sample)
end_time = time.time()
print(f"t-SNE 소요 시간: {end_time - start_time:.2f} 초")

def plot_2d(X, y, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# t-SNE 결과 시각화
plot_2d(X_tsne, y_sample, "t-SNE 2D Visualization")

## 특징벡터로 tsne 비선형 차원축소로 라벨2개이기에 , 1280 길이인 특징벡터를 길이가 2인 벡터로 줄여서 라벨의 분포를 확인해도 제대로 군집이 되어있지 않음. 
## 그래서 아래와 같은 accuracy 자체가 70% 정도로 낮은 값을 가지는 것으로 확인된다. 


# In[117]:


import matplotlib.pyplot as plt

# 에포크마다의 loss와 accuracy 데이터
epochs = [1, 2, 3, 4, 5]
loss = [0.663, 0.658, 0.649, 0.640, 0.632]
accuracy = [62.24, 62.44, 62.67, 63.04, 63.34]

# 그래프 크기 설정
plt.figure(figsize=(10, 6))

# 첫 번째 y축 (Loss)
fig, ax1 = plt.subplots()

ax1.plot(epochs, loss, marker='o', color='skyblue', label='Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_title('Loss and Accuracy over Epochs')
ax1.grid(True)

# 두 번째 y축 (Accuracy)
ax2 = ax1.twinx()
ax2.plot(epochs, accuracy, marker='o', color='salmon', label='Accuracy')
ax2.set_ylabel('Accuracy (%)', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')

# 범례 추가
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

# 그래프 표시
plt.show()


# #### 기본적으로 조건을 따지지 않고 , 시퀀스 하지않고 개별데이터로 lstm layer만 추가해서 이상행동 분류 모델링 진행 (이미지 데이터 기준)

# In[19]:


# 0.62? 정도 밖에 성능이 안나옴 :> npy 파일로 전처리된 데이터의 성능 
# 원본 데이터의 성능 : 0.8  / val_accuracy : 0.78    역시 원본데이터 자체를 건드는게 훨씬 낫다. 

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 첫 번째 GPU만 사용하도록 설정
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)
else:
    print("사용 가능한 GPU가 없습니다.")
    
X_img_train , X_img_val, y_img_train, y_img_val = train_test_split(X_img, y_img , test_size = 0.2 ,  random_state =42)

model = Sequential()
model.add(LSTM(128, input_shape=(1,X_img.shape[1]), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

X_img_train_reshaped = np.expand_dims(X_img_train, axis=1)
X_img_val_reshaped = np.expand_dims(X_img_val, axis=1)

model.fit(X_img_train_reshaped , y_img_train, epochs=10 , batch_size=16, validation_data = (X_img_val_reshaped,y_img_val))



# In[15]:


# dropout (0.3), batchnormalization , dense층 형성  
# 전처리 데이터로 특징벡터 추출한 데이터로 학습한 결과 : 0.62
# val_accuracy : 0.79  원본데이터 기준.
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
X_img_train , X_img_val, y_img_train, y_img_val = train_test_split(X_img, y_img , test_size = 0.2 ,  random_state =42)

model = Sequential()
model.add(LSTM(256, input_shape=(1, X_img.shape[1]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_img_train_reshaped = np.expand_dims(X_img_train, axis=1)
X_img_val_reshaped = np.expand_dims(X_img_val, axis=1)

model.fit(X_img_train_reshaped, y_img_train, epochs=20, batch_size=32, validation_data=(X_img_val_reshaped,y_img_val))
val_loss, val_accuracy = model.evaluate(X_img_val_reshaped, y_img_val)
print(f'Validation Accuracy: {val_accuracy}')


# In[ ]:





# #### 데이터 증강 , 및 스케일링 진행 

# In[20]:


# 시퀀스를 처리하지 않고 , 개별 이미지로 진행한 결과 75%를 넘지 않는다. 
# 딱히 큰 의미가 있기에는 무리가 있다.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from imblearn.over_sampling import SMOTE

# 데이터 분할
X_img_train, X_img_val, y_img_train, y_img_val = train_test_split(X_img, y_img, test_size=0.2, random_state=42)

# 스케일러와 옵티마이저 리스트 정의
scalers = [StandardScaler(), RobustScaler(), MinMaxScaler()]
scaler_names = ['StandardScaler', 'RobustScaler', 'MinMaxScaler']
optimizers = [Adam(), SGD(), RMSprop()]
optimizer_names = ['Adam', 'SGD', 'RMSprop']

# 결과 저장을 위한 리스트
results = []

for scaler, scaler_name in zip(scalers, scaler_names):
    # 데이터 정규화
    X_img_train_scaled = scaler.fit_transform(X_img_train)
    X_img_val_scaled = scaler.transform(X_img_val)

    # 데이터 3차원 변환
    X_img_train_reshaped = np.expand_dims(X_img_train_scaled, axis=1) # 3차원 형태로 맞춤 , (samples, timesteps, features)
    X_img_val_reshaped = np.expand_dims(X_img_val_scaled, axis=1)

    # SMOTE 오버샘플링
    smote = SMOTE(random_state=42)
    X_img_train_resampled, y_img_train_resampled = smote.fit_resample(X_img_train_reshaped.reshape(-1, X_img_train_reshaped.shape[-1]), y_img_train)
    X_img_train_resampled = np.expand_dims(X_img_train_resampled, axis=1)

    for optimizer, optimizer_name in zip(optimizers, optimizer_names):
        # 모델 정의
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, input_shape=(1, X_img.shape[1])))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))

        # 모델 컴파일
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # 모델 학습
        model.fit(X_img_train_resampled, y_img_train_resampled, epochs=5, batch_size=128, validation_data=(X_img_val_reshaped, y_img_val), verbose=0)

        # 모델 평가
        val_loss, val_accuracy = model.evaluate(X_img_val_reshaped, y_img_val, verbose=0)
        results.append((scaler_name, optimizer_name, val_accuracy))
        print(f'Scaler: {scaler_name}, Optimizer: {optimizer_name}, Validation Accuracy: {val_accuracy}')

# 결과 출력
for result in results:
    print(f'Scaler: {result[0]}, Optimizer: {result[1]}, Validation Accuracy: {result[2]}')


# In[19]:


pip install imbalanced-learn==0.12.3


# #### pytorch 사용 모델링 

# In[8]:


# # pytorch 로 정규화를 진행해서 모델링을 돌리니 70% 정도의 accuracy를 가진다.
# import torch
# import numpy as np
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler

# # 데이터 분할
# X_train, X_val, y_train, y_val = train_test_split(X_img, y_img, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

# # NumPy 배열을 PyTorch 텐서로 변환
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)  # Numpy 배열로 변환
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)  # Numpy 배열로 변환
# #torch.long은 

# # 데이터셋 및 데이터로더 생성
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# import torch.nn as nn
# import torch.optim as optim

# class FCNN(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(FCNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 1024)
#         self.dropout1 = nn.Dropout(0.5)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.dropout2 = nn.Dropout(0.5)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 256)
#         self.dropout3 = nn.Dropout(0.5)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.fc4 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.dropout1(x)
#         x = self.bn1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = self.dropout2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         x = self.dropout3(x)
#         x = self.bn3(x)
#         x = torch.relu(x)
#         x = self.fc4(x)
#         return x

# # 모델 인스턴스 생성
# input_dim = X_train.shape[1]  # 특징 벡터의 차원
# num_classes = len(np.unique(y_img))  # 클래스 수
# model = FCNN(input_dim, num_classes)

# # 손실 함수 및 옵티마이저 정의
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # CUDA 사용 가능 여부 확인
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 조기 종료 설정
# best_val_loss = float('inf')
# patience = 5
# trigger_times = 0

# # 모델 학습
# num_epochs = 30
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
    
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
#     # Validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
    
#     val_loss /= len(val_loader)
#     print(f"Validation Loss: {val_loss}")
    
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         trigger_times = 0
#         torch.save(model.state_dict(), 'best_model.pth')
#     else:
#         trigger_times += 1
#         if trigger_times >= patience:
#             print('Early stopping!')
#             break

# # Load the best model
# model.load_state_dict(torch.load('best_model.pth'))

# # 모델 평가
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in val_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Validation Accuracy: {100 * correct / total}%')

# Epoch [1/30], Loss: 0.6561252077776136
# Validation Loss: 0.6491673351151643
# Epoch [2/30], Loss: 0.6379304706872532
# Validation Loss: 0.6489408958789914
# Epoch [3/30], Loss: 0.6282461281546365
# Validation Loss: 0.6360193621422752
# Epoch [4/30], Loss: 0.6213306915528911
# Validation Loss: 0.6270193057040337
# Epoch [5/30], Loss: 0.6146526821774391
# Validation Loss: 0.6438208436887749
# Epoch [6/30], Loss: 0.6102134921717874
# Validation Loss: 0.6018935418157402
# Epoch [7/30], Loss: 0.6056043793868876
# Validation Loss: 0.6329060042800119
# Epoch [8/30], Loss: 0.6006664102865101
# Validation Loss: 0.5895707567639484
# Epoch [9/30], Loss: 0.5976796289729503
# Validation Loss: 0.5957320837229579
# Epoch [10/30], Loss: 0.5938450277651717
# Validation Loss: 0.5802458484541041
# Epoch [11/30], Loss: 0.5910814177181007
# Validation Loss: 0.6370570487256696
# Epoch [12/30], Loss: 0.5882271704746662
# Validation Loss: 0.634806594467618
# Epoch [13/30], Loss: 0.5845621274043872
# ...
# Validation Loss: 0.57949686533792
# Epoch [23/30], Loss: 0.5642383067088728
# Validation Loss: 0.595008215594306
# Early stopping!


# In[18]:


#SMOTE를 사용해서 라벨의 균형을 맞추고 진행을 한다.. 그래도 특징벡터이기에 성능이 저조할 수 있음.
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from tqdm import tqdm  # tqdm 라이브러리 추가
from sklearn.preprocessing import StandardScaler

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_img, y_img, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)

# SMOTE를 사용한 오버샘플링
smote = SMOTE(sampling_strategy={0: 160000, 1: 160000}, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# NumPy 배열을 PyTorch 텐서로 변환
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)

# 데이터셋 및 데이터로더 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.optim as optim

class FCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

# 모델 인스턴스 생성
input_dim = X_train.shape[1]  # 특징 벡터의 차원
num_classes = len(np.unique(y_img))  # 클래스 수
model = FCNN(input_dim, num_classes)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.is_available():
    print(f"CUDA is available. PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")

# 모델 학습
num_epochs = 3
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # 모델 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy}%')

    # 최상의 검증 성능을 가진 모델 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

print(f'Best Validation Accuracy: {best_val_accuracy}%')


# ### 모델링 방안2 , 전처리된 이미지 데이터로 전이학습 실시 

# #### 이미지데이터를 시퀀스 처리 후 모델 

# In[6]:


#이미지 시퀀스처리 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sample_size = int(1.0 * len(df_target))
sample_indices = np.arange(sample_size)

# 이미지 경로와 라벨 추출
image_paths = df_target['img_file'].values[sample_indices]
labels = df_target['label'].values[sample_indices]


# 이미지를 5개씩 시퀀스로 그룹화
sequence_length = 5
num_sequences = len(image_paths) // sequence_length

image_sequences = [image_paths[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)]
label_sequences = [labels[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)]


# In[57]:


# npy 파일로 전처리된 데이터로 학습시킨 결과임 ! 그래도 80% 정도 나온다 
#  전체 데이터셋의 10%만 테스트 용으로 돌린 결과임
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class EfficientNetWithLSTM(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithLSTM, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        self.efficientnet.classifier = nn.Identity()  # 기존 분류 레이어 제거
        
        # 추가 CNN 레이어
        self.additional_cnn = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        
        # Fully Connected 레이어
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # 배치와 시퀀스 차원을 평탄화
        x = self.efficientnet.features(x)  # EfficientNet의 특징 추출기 부분만 사용
        x = self.additional_cnn(x)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, features)로 다시 변형
        x, _ = self.lstm(x)  # LSTM을 통해 시퀀스 처리
        x = x[:, -1, :]  # 마지막 LSTM 출력만 사용
        x = self.fc(x)
        return x

class ImageSequenceDataset(Dataset):
    def __init__(self, image_sequences, label_sequences):
        self.image_sequences = image_sequences
        self.label_sequences = label_sequences

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        images = [np.load(img_path) for img_path in self.image_sequences[idx]]
        labels = self.label_sequences[idx]
        
        # 이미지 배열을 (H, W, C)에서 (C, H, W)로 변환
        images = [torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in images]
        label = torch.tensor(labels[0], dtype=torch.long)  # 첫 번째 이미지의 라벨 사용
        
        return torch.stack(images), label

# 데이터셋 인스턴스 생성
dataset = ImageSequenceDataset(image_sequences, label_sequences)

# 전체 데이터셋의 10%만 사용
subset_size = int(0.1 * len(dataset))
subset_indices = np.arange(subset_size)
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

# 데이터 분할
train_indices, val_indices = train_test_split(np.arange(len(subset_dataset)), test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(subset_dataset, train_indices)
val_dataset = torch.utils.data.Subset(subset_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 모델 인스턴스 생성
num_classes = 2  # 클래스 수 (0과 1)
model = EfficientNetWithLSTM(num_classes)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 및 평가 루프
num_epochs = 3
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # 모델 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy}%')

    # 최상의 검증 성능을 가진 모델 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        #torch.save(model.state_dict(), 'best_model_img_process_cnn.pth')

print(f'Best Validation Accuracy: {best_val_accuracy}%')


# In[11]:


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class EfficientNetWithLSTM(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithLSTM, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        self.efficientnet.classifier = nn.Identity()  # 기존 분류 레이어 제거
        
        # 추가 CNN 레이어
        self.additional_cnn = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        
        # Fully Connected 레이어
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # 배치와 시퀀스 차원을 평탄화
        x = self.efficientnet.features(x)  # EfficientNet의 특징 추출기 부분만 사용
        x = self.additional_cnn(x)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, features)로 다시 변형
        x, _ = self.lstm(x)  # LSTM을 통해 시퀀스 처리
        x = x[:, -1, :]  # 마지막 LSTM 출력만 사용
        x = self.fc(x)
        return x

class ImageSequenceDataset(Dataset):
    def __init__(self, image_sequences, label_sequences, transform=None):
        self.image_sequences = image_sequences
        self.label_sequences = label_sequences
        self.transform = transform

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        images = []
        for img_path in self.image_sequences[idx]:
            try:
                img = Image.open(img_path).convert('RGB')  # 이미지 로드 및 RGB로 변환
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        if not images:
            raise ValueError(f"No valid images found for index {idx}")
        
        labels = self.label_sequences[idx]
        label = torch.tensor(labels[0], dtype=torch.long)  # 첫 번째 이미지의 라벨 사용
        
        return torch.stack(images), label

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 데이터셋 인스턴스 생성
dataset = ImageSequenceDataset(image_sequences, label_sequences, transform=transform)

# 전체 데이터셋의 10%만 사용
subset_size = int(0.15 * len(dataset))
subset_indices = np.arange(subset_size)
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

# 데이터 분할
train_indices, val_indices = train_test_split(np.arange(len(subset_dataset)), test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(subset_dataset, train_indices)
val_dataset = torch.utils.data.Subset(subset_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 인스턴스 생성
num_classes = 2  # 클래스 수 (0과 1)
model = EfficientNetWithLSTM(num_classes)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 및 평가 루프
num_epochs = 5
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # 모델 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy}%')

    # 최상의 검증 성능을 가진 모델 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model_img_process_cnn_.pth')

print(f'Best Validation Accuracy: {best_val_accuracy}%')


# In[10]:


torch.save(model.state_dict(), 'best_model_img_process_lstm_CNN.pth')


# In[7]:


model = EfficientNetWithLSTM(num_classes)  # 모델 구조 정의
model.load_state_dict(torch.load('best_model_img_process_lstm_CNN.pth'))
model.eval()  # 평가 모드로 전환


# In[8]:


torch.save(model, 'best_model_img_process_lstm_CNN_full.pth')


# In[9]:


model = torch.load('best_model_img_process_lstm_CNN_full.pth')
model.eval()  # 평가 모드로 전환


# In[ ]:


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class EfficientNetWithLSTM(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithLSTM, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        self.efficientnet.classifier = nn.Identity()  # 기존 분류 레이어 제거
        
        # 추가 CNN 레이어
        self.additional_cnn = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        
        # Fully Connected 레이어
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # 배치와 시퀀스 차원을 평탄화
        x = self.efficientnet.features(x)  # EfficientNet의 특징 추출기 부분만 사용
        x = self.additional_cnn(x)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, features)로 다시 변형
        x, _ = self.lstm(x)  # LSTM을 통해 시퀀스 처리
        x = x[:, -1, :]  # 마지막 LSTM 출력만 사용
        x = self.fc(x)
        return x

class ImageSequenceDataset(Dataset):
    def __init__(self, image_sequences, label_sequences, transform=None):
        self.image_sequences = image_sequences
        self.label_sequences = label_sequences
        self.transform = transform

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        images = []
        for img_path in self.image_sequences[idx]:
            try:
                img = Image.open(img_path).convert('RGB')  # 이미지 로드 및 RGB로 변환
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        if not images:
            raise ValueError(f"No valid images found for index {idx}")
        
        labels = self.label_sequences[idx]
        label = torch.tensor(labels[0], dtype=torch.long)  # 첫 번째 이미지의 라벨 사용
        
        return torch.stack(images), label

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 데이터셋 인스턴스 생성
dataset = ImageSequenceDataset(image_sequences, label_sequences, transform=transform)

# 전체 데이터셋의 10%만 사용
subset_size = int(0.1 * len(dataset))
subset_indices = np.arange(subset_size)
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

# 데이터 분할
train_indices, val_indices = train_test_split(np.arange(len(subset_dataset)), test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(subset_dataset, train_indices)
val_dataset = torch.utils.data.Subset(subset_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 인스턴스 생성
num_classes = 2  # 클래스 수 (0과 1)
model = EfficientNetWithLSTM(num_classes)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 및 평가 루프
num_epochs = 20
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # 모델 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy}%')

    # 최상의 검증 성능을 가진 모델 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model_img_process_cnn_.pth')

print(f'Best Validation Accuracy: {best_val_accuracy}%')


# ### 영상데이터만으로 전처리 후 모델링 

# In[91]:


df_video = pd.read_csv("df_video_labeled.csv")
#df_video.loc[df_video['frame_count']==0]  # 프레임 0이 존재하지 않는데?
df_video.head()


# In[52]:


import matplotlib.pyplot as plt

# 한국어 폰트 설정
plt.rc('font', family='Malgun Gothic')

# 데이터 준비
category_counts = df_video['category_name'].value_counts()

# 그래프 그리기
plt.figure(figsize=(10, 6))
bars = plt.bar(category_counts.index, category_counts.values)

# 각 막대 위에 수치 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # va='bottom'은 텍스트를 막대 위에 표시

# 그래프 제목 및 축 레이블 설정
plt.title('카테고리별 분류')
plt.xlabel('카테고리')
plt.ylabel('비디오 수')

# 그래프 출력
plt.show()


# In[80]:


label = {'졸음운전':0, '음주운전':0, '물건찾기':0 , '통화':0, '휴대폰조작':0, '차량제어':1, '운전자폭행':0}
df_video['label'] = df_video['category_name'].map(label)
df_video['label'].value_counts()
df_video

df_video.to_csv('df_video_labeld.csv', index=False)


# In[94]:


df_sample = df_video.head(100)
df_sample


# In[111]:


# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# import pandas as pd
# from tqdm import tqdm

# # 이미지 전처리 정의
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def preprocess_frame(frame):
#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     img = transform(img)
#     return img

# def average_frames(frames):
#     frames_array = np.array([frame.numpy() for frame in frames])
#     avg_frame = np.mean(frames_array, axis=0)
#     avg_frame_tensor = torch.tensor(avg_frame, dtype=torch.float32)
#     return avg_frame_tensor

# def adjust_frame_sequence(frames, target_length):
#     num_frames = len(frames)

#     if num_frames > target_length:
#         group_size = num_frames // target_length
#         averaged_frames = []
#         for i in range(0, num_frames, group_size):
#             group = frames[i:i + group_size]
#             avg_frame = average_frames(group)
#             averaged_frames.append(avg_frame)
#         frames = averaged_frames[:target_length]
    
#     elif num_frames < target_length:
#         repeat_count = target_length // num_frames
#         remainder = target_length % num_frames
#         frames = frames * repeat_count + frames[:remainder]
    
#     return frames

# def extract_and_preprocess_frames(video_path, target_length):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video file: {video_path}")
    
#     frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break 
#         frames.append(preprocess_frame(frame))
#     cap.release()

#     frames = adjust_frame_sequence(frames, target_length)
#     return torch.stack(frames)


# # 고정할 프레임 수 설정 (중간값 사용)
# target_length = int(df_sample['frame_count'].median())

# # tqdm을 사용하여 진행 상황 표시
# processed_frames = []
# for video_file in tqdm(df_sample['video_file'].head(100), desc="Processing videos"):
#     try:
#         processed_frames.append(extract_and_preprocess_frames(video_file, target_length))
#     except FileNotFoundError as e:
#         print(e)
#         processed_frames.append(None)  # 처리할 수 없는 비디오 파일에 대해 None 추가

# # 결과를 데이터프레임에 저장
# df_sample['processed_frames'] = processed_frames

# # 결과 출력
# df_sample
# #Unable to allocate 184. MiB for an array with shape (160, 3, 224, 224) and data type object


# In[114]:


import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

# GPU 메모리 캐시 비우기
torch.cuda.empty_cache()

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img)
    return img

def average_frames(frames):
    frames_array = np.array([frame.numpy() for frame in frames])
    avg_frame = np.mean(frames_array, axis=0)
    avg_frame_tensor = torch.tensor(avg_frame, dtype=torch.float32)
    return avg_frame_tensor

def adjust_frame_sequence(frames, target_length):
    num_frames = len(frames)

    if num_frames > target_length:
        group_size = num_frames // target_length
        averaged_frames = []
        for i in range(0, num_frames, group_size):
            group = frames[i:i + group_size]
            avg_frame = average_frames(group)
            averaged_frames.append(avg_frame)
        frames = averaged_frames[:target_length]
    
    elif num_frames < target_length:
        repeat_count = target_length // num_frames
        remainder = target_length % num_frames
        frames = frames * repeat_count + frames[:remainder]
    
    return frames

def extract_and_preprocess_frames(video_path, target_length):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        frames.append(preprocess_frame(frame))
    cap.release()

    frames = adjust_frame_sequence(frames, target_length)
    return torch.stack(frames)

# 고정할 프레임 수 설정 (중간값 사용)
target_length = int(df_sample['frame_count'].median())

# 중간 결과 저장 디렉토리 생성
output_dir = 'processed_frames'
os.makedirs(output_dir, exist_ok=True)

# tqdm을 사용하여 진행 상황 표시
for idx, video_file in tqdm(enumerate(df_video['video_file']), desc="Processing videos", total=len(df_video['video_file'])):
    try:
        frames = extract_and_preprocess_frames(video_file, target_length)
        torch.save(frames, os.path.join(output_dir, f'frames_{idx}.pt'))
    except FileNotFoundError as e:
        print(e)
        torch.save(None, os.path.join(output_dir, f'frames_{idx}.pt'))  # 처리할 수 없는 비디오 파일에 대해 None 저장

# 결과를 데이터프레임에 저장
df_video['processed_frames'] = [os.path.join(output_dir, f'frames_{idx}.pt') for idx in range(len(df_video['video_file']))]

# 결과 출력
df_video.to_csv("get_tensor.csv", index=False)
df_video.head()


# In[101]:


df_sample


# In[103]:


import torch

# 저장된 텐서 파일을 로드하는 함수
def load_processed_frames(file_path):
    return torch.load(file_path)

# 예제 파일 경로
file_path = 'processed_frames/frames_0.pt'

# 텐서 로드
tensor = load_processed_frames(file_path)

# 텐서 형태 확인
print(f"Tensor shape: {tensor.shape}")

# 텐서 내용 확인 (일부만 출력)
print(tensor)

df_sample['frame_count'].median()


# In[110]:


import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# 저장된 텐서 파일을 로드하는 함수
def load_processed_frames(file_path):
    return torch.load(file_path)

# 텐서에서 이미지로 변환하는 함수
def tensor_to_image(tensor, index):
    # 텐서에서 특정 프레임을 선택
    frame_tensor = tensor[index]
    # 텐서를 PIL 이미지로 변환
    transform = transforms.ToPILImage()
    img = transform(frame_tensor)
    return img

# 예제 파일 경로
file_path = 'processed_frames/frames_98.pt'

# 텐서 로드
tensor = load_processed_frames(file_path)

# 텐서 형태 확인
print(f"Tensor shape: {tensor.shape}")

# 텐서의 첫 번째 프레임을 이미지로 변환
img = tensor_to_image(tensor, 10)

# 이미지 출력
img.show()


# In[ ]:




