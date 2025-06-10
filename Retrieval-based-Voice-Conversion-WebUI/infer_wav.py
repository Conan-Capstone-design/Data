import os
import glob
import torch
import numpy as np
import soundfile as sf
import librosa

from infer.modules.vc.modules import VC  # VC 클래스
from conan_fine.config import Config     # 네가 정의한 설정

# ===== 설정 =====
config = Config()
vc = VC(config)

# ===== 모델 로딩 =====
sid_path = "conan_fine/conan-fine.pth"  # 네가 학습한 모델 경로
vc.get_vc(sid_path)  # 내부적으로 모델과 config 로딩

# ===== 입력/출력 경로 =====
input_dir = "wavefile/input"
output_dir = "wavefile/output"
os.makedirs(output_dir, exist_ok=True)

# ===== .wav 파일 목록 가져오기 =====
wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
if not wav_files:
    print(f"'{input_dir}'에 .wav 파일이 없습니다.")
    exit()

# ===== 추론 루프 =====
for wav_path in wav_files:
    filename = os.path.basename(wav_path)
    print(f"변환 중: {filename}")

    # 오디오 로딩 및 tmp 저장 (VC는 경로 기반 입력만 받음)
    audio, sr = librosa.load(wav_path, sr=16000)
    sf.write("tmp.wav", audio, 16000)

    # 추론 실행
    info, (tgt_sr, output_audio) = vc.vc_single(
        sid=0,                      # 다인 모델이면 spk_id 지정, 단일이면 0
        input_audio_path="tmp.wav",
        f0_up_key=0,                # 피치 조절값 (0 = 원음, 12 = 한 옥타브 위)
        f0_file=None,               # 커스텀 f0 사용 안 함
        f0_method="rmvpe",          # pm / harvest / rmvpe 중 선택
        file_index="",              # 인덱스 안 씀
        file_index2="",
        index_rate=0.75,
        filter_radius=3,
        resample_sr=0,              # 출력 샘플레이트 (0 = 원본 유지)
        rms_mix_rate=0.25,
        protect=0.33,
    )

    # 출력 저장
    output_path = os.path.join(output_dir, filename)
    sf.write(output_path, output_audio, tgt_sr)
    print(f"저장 완료: {output_path}")

print("전체 변환 완료!")
