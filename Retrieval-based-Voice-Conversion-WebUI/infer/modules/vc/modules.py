import os
import time
import math
import torch
import logging
import traceback
import numpy as np
import soundfile as sf
import parselmouth
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)



class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.hubert_model = None
        self.config = config
        self.config.device = torch.device("cuda:5")

    def get_mean_f0(self, wav_path):
        snd = parselmouth.Sound(str(wav_path))
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array['frequency']
        return np.mean(f0_values[f0_values > 0])

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if self.hubert_model is not None:
                logger.info("Clean model cache")
                del self.net_g, self.n_spk, self.hubert_model, self.tgt_sr
                self.hubert_model = self.net_g = self.n_spk = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {"visible": True, "value": to_return_protect0, "__type__": "update"},
                {"visible": True, "value": to_return_protect1, "__type__": "update"},
                "",
                "",
            )

        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None

        if self.pipeline is None:
            self.pipeline = Pipeline(self.tgt_sr, self.config)

        if f0_up_key in [None, "", "auto"]:
            speaker_f0 = self.get_mean_f0(input_audio_path)
            TARGET_F0 = 260.3701554543574
            f0_up_key = round(math.log2(TARGET_F0 / speaker_f0) * 12 - 1.29)
            logger.info(f"[Auto Transpose] {input_audio_path} 평균 f0: {round(speaker_f0,2)} Hz → f0_up_key: {f0_up_key}")
        else:
            f0_up_key = int(f0_up_key)

        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip().strip('"').strip("\n").replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )

            tgt_sr = resample_sr if self.tgt_sr != resample_sr >= 16000 else self.tgt_sr

            return "Success", (tgt_sr, audio_opt)

        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = dir_path.strip().strip('"').strip("\n")
            opt_root = opt_root.strip().strip('"').strip("\n")

            all_infos = []

            for root, _, files in os.walk(dir_path):
                rel_path = os.path.relpath(root, dir_path)
                cur_output_dir = os.path.join(opt_root, rel_path)
                os.makedirs(cur_output_dir, exist_ok=True)

                wav_paths = [os.path.join(root, f) for f in files if f.lower().endswith(".wav")]
                if not wav_paths:
                    continue

                try:
                    sample_f0 = self.get_mean_f0(wav_paths[0])
                    ### 타겟 화자의 평균 F0
                    TARGET_F0 = 260.3701554543574 
                    local_f0_up_key = round(math.log2(TARGET_F0 / sample_f0) * 12) - 1.29
                    logger.info(f"[{rel_path}] 평균 f0: {round(sample_f0, 2)} Hz → f0_up_key: {local_f0_up_key}")
                except Exception as e:
                    local_f0_up_key = 0
                    logger.warning(f"[{rel_path}] f0 계산 실패, f0_up_key=0 적용: {e}")

                def process(path):
                    out_path = os.path.join(
                        cur_output_dir,
                        os.path.splitext(os.path.basename(path))[0] + f".{format1}",
                    )
                    if os.path.exists(out_path):
                        return f"{os.path.basename(path)} -> 이미 존재함, 스킵"

                    for attempt in range(2):
                        try:
                            info, opt = self.vc_single(
                                sid,
                                path,
                                local_f0_up_key,
                                None,
                                f0_method,
                                file_index,
                                file_index2,
                                index_rate,
                                filter_radius,
                                resample_sr,
                                rms_mix_rate,
                                protect,
                            )
                            if "Success" in info:
                                tgt_sr, audio_opt = opt
                                if format1 in ["wav", "flac"]:
                                    sf.write(out_path, audio_opt, tgt_sr)
                                else:
                                    with BytesIO() as wavf:
                                        sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                        wavf.seek(0)
                                        with open(out_path, "wb") as outf:
                                            wav2(wavf, outf, format1)
                                return f"{os.path.basename(path)} -> {info}"
                            else:
                                raise Exception(info)
                        except Exception as e:
                            if attempt == 0:
                                time.sleep(5)
                            else:
                                return f"{os.path.basename(path)} 변환 실패: {str(e)}"

                with ThreadPoolExecutor(max_workers=50) as executor:
                    future_to_path = {executor.submit(process, path): path for path in wav_paths}
                    for future in as_completed(future_to_path):
                        result = future.result()
                        all_infos.append(result)
                        yield result

                print(f"{rel_path} 폴더 변환 완료")

            print("전체 변환 완료")

        except Exception:
            yield traceback.format_exc()
