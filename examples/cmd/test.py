import os, sys
import ChatTTS
from tools.audio import wav_arr_to_mp3_view
from tools.logger import get_logger
import torch
import torchaudio
import pybase16384 as b14
import numpy as np
import lzma
from pydub import AudioSegment


if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

logger = get_logger("Command")


# 压缩方法，将tensor压缩并使用pybase16384编码压缩后tensor为字符串
def compress_and_encode(tensor):
    np_array = tensor.numpy().astype(np.float16)
    compressed = lzma.compress(np_array.tobytes(), format=lzma.FORMAT_RAW,
                               filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    encoded = b14.encode_to_string(compressed)
    return encoded

# 将音频文件保存为mp3
def save_mp3_file(wav, name: str):
    data = wav_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{name}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"mp3音频文件已保存：{mp3_filename}")

# wav格式 转 mp3
def wav_to_mp3(wav_path, mp3_path):
    # 加载 WAV 文件
    audio = AudioSegment.from_wav(wav_path)
    # 将 WAV 文件导出为 MP3 格式
    audio.export(mp3_path, format="mp3")
    print(f"转换成功：{wav_path} -> {mp3_path}")



def main(texts: list[str], output_name: str, seed_num: int):
    logger.info("输入文本: %s", str(texts))

    chat = ChatTTS.Chat(get_logger("ChatTTS"))

    if chat.load():
        logger.info("ChatTTS 加载成功")
    else:
        logger.error("ChatTTS 加载失败")
        sys.exit(1)


    # Refine text :是否对文本进行口语化处理

    # Text Seed :配置文本种子值，不同种子对应不同口语化风格

    # Output Text : 口语化处理后生成的文本


    # 音色控制：可以修改男声、女声
    # Timbre : 预设的音色种子值
    # Audio Seed : 配置音色种子值，不同种子对应不同音色
    # Speaker Embedding : 音色码

    # 获取随机音色
    # spk_emb_str = chat.sample_random_speaker()
    # 获取资源中的音色
    seed_emb = "asset/seed_" + str(seed_num) + "_restored_emb.pt"
    spk = torch.load(seed_emb, map_location=torch.device('cpu')).detach()
    spk_emb_str = compress_and_encode(spk)
    print(f"音色码：{spk_emb_str}")

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = spk_emb_str,      # 音色
        temperature = .3,   # 控制音频情感波动性，范围为 0-1，数字越大，波动性越大
        top_P = 0.7,        # 控制音频的情感相关性，范围为 0.1-0.9，数字越大，相关性越高
        top_K = 20,         # 控制音频的情感相似性，范围为 1-20，数字越小，相似性越高
    )

    # oral_(0-9), laugh_(0-2), break_(0-7)
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
    )


    wavs = chat.infer(
        texts,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )

    logger.info("ChatTTS 翻译完成")

    # 输出wav格式文件
    # torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)

    # 将每个生成的wav文件保存到本地 mp3 文件
    for index, wav in enumerate(wavs):
        name = str(index)
        if len(output_name) > 0 :
            name = output_name
        save_mp3_file(wav, name)
    logger.info("ChatTTS 音频生成成功")


if __name__ == "__main__":
    logger.info("开始执行，本文件代码")

    # 情绪控制功能
    # [uv_break]、[lbreak]
    # [laugh]

    # 音色下载：https://huggingface.co/spaces/taa/ChatTTS_Speaker
    # 男声：402、1019、1128、729、
    # 女声：1983、1910、1363、1518、181、1528、1397、2310、1519、1089、1096
    seed_num = 1096

    

    texts = ["那大家总要有个精神寄托，以前是生娃 然后游戏机 电脑 手机 ，小孩有小孩的 大人还应该有大人，[laugh] 可以，要不生产AI故事 挂到知乎 这些地方 看看有没有人读，这样吗，[laugh]，你要不要找个好听的声音，[laugh]，我的不好听，还不如你的"]
    main(texts, str(seed_num), seed_num)
    logger.info("本文件代码，执行完成")