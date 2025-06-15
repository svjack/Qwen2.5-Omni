import os
import re
import soundfile as sf
import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


# 加载模型
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

# 设置输入和输出目录
input_dir = "wang_generated_ads"
output_dir = "mavuika_generated_ads"
os.makedirs(output_dir, exist_ok=True)

# 正则表达式过滤中文字符
def filter_chinese_extended(text):
    extended_pattern = re.compile(r'[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]')
    return ''.join(extended_pattern.findall(text))

# 遍历指定文件夹下的所有 txt 文件，并按字典序排序
txt_files = sorted(
    [f for f in os.listdir(input_dir) if f.endswith(".txt")],
    key=lambda x: x  # 按字典序排序
)

# 设置是否使用视频中的音频
USE_AUDIO_IN_VIDEO = False

# 遍历所有 txt 文件
for idx, filename in enumerate(txt_files):
    file_path = os.path.join(input_dir, filename)
    
    # 读取广告文本
    with open(file_path, "r", encoding="utf-8") as f:
        ad_text = f.read().strip()
    
    # 过滤中文字符
    ad_text = filter_chinese_extended(ad_text)
    
    print(f"Processing file {filename}...")    
    print("Generated ad text:", ad_text)

    # 构造新对话用于生成语音
    voice_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"读出下面这个广告：{ad_text}"}
            ],
        },
    ]

    # 准备语音合成输入
    voice_text = processor.apply_chat_template(voice_conversation, add_generation_prompt=True, tokenize=False)
    v_audios, v_images, v_videos = process_mm_info(voice_conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    v_inputs = processor(text=voice_text, audio=v_audios, images=v_images, videos=v_videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    v_inputs = v_inputs.to(model.device).to(model.dtype)

    # 生成语音
    voice_ids, audio = model.generate(**v_inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=True, speaker="Chelsie")
    audio_np = audio.reshape(-1).detach().cpu().numpy()

    # 定义文件名
    filename_base = os.path.splitext(filename)[0]  # 去除 .txt 后缀

    # 复制原始 txt 文件到输出目录
    output_txt_path = os.path.join(output_dir, f"{filename_base}.txt")
    os.system(f"cp {file_path} {output_txt_path}")

    # 保存生成的 wav 文件
    wav_path = os.path.join(output_dir, f"{filename_base}.wav")
    sf.write(wav_path, audio_np, samplerate=24000)

    print(f"Saved to {output_txt_path} and {wav_path}")

