import os
import soundfile as sf
import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from datasets import load_dataset

# 加载模型
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

# 加载数据集
ds = load_dataset("svjack/InfiniteYou_PosterCraft_Wang_Leehom_Poster_FP8")["train"]

# 创建输出目录
output_dir = "wang_generated_ads"
os.makedirs(output_dir, exist_ok=True)

import re
def filter_chinese_characters(text):
    """过滤字符串，仅保留中文字符（包括常用汉字和中文标点符号）
    
    参数:
        text (str): 输入的字符串，可能包含中文、英文、数字等混合内容
        
    返回:
        str: 仅包含中文字符的字符串
    """
    # 方法1：使用正则表达式匹配中文字符（基本汉字范围）
    pattern = re.compile(r'[\u4e00-\u9fa5]')  # 匹配基本汉字[1,2,6,8](@ref)
    chinese_only = ''.join(pattern.findall(text))
    
    return chinese_only

def filter_chinese_extended(text):
    """过滤字符串，保留中文字符及中文标点符号
    
    参数:
        text (str): 输入的字符串
        
    返回:
        str: 包含中文及中文标点的字符串
    """
    # 扩展正则表达式，包含中文标点符号（如，。、；：等）[3,8](@ref)
    extended_pattern = re.compile(r'[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]')
    return ''.join(extended_pattern.findall(text))

# 设置是否使用视频中的音频
USE_AUDIO_IN_VIDEO = False

# 遍历数据集
for idx, item in enumerate(ds):
    print(f"Processing sample {idx}...")

    # 构建系统提示
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": '''
                你是一个广告营销专家，现在给你一个对于某商品的英文描述及对应的英文海报描述，
                要求你给出其合适的一句话的英文广告词。直接读出这个广告词作为回复，不要参杂任何分析。
                下面会提供给你关于这个广告的信息。
                注意：你生成的广告词必须为全部中文，不得包含任何英文。
                '''}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": '''
                商品类别为：{}
                海报种类：{}
                海报设计：{}
                注意：你生成的广告词必须为全部中文，不得包含任何英文。
                '''.format(item["product_category"], item["poster_prompt"], item["final_prompt"])
                }
            ],
        },
    ]

    # 准备输入
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    # 生成广告词
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
    text_output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ad_text = text_output[0].split("assistant")[-1].strip()
    if bool(filter_chinese_characters(ad_text).strip()):
        ad_text = filter_chinese_extended(ad_text)
    ad_text = "，".join(ad_text.split("，")[:4])
    
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
    voice_ids, audio = model.generate(**v_inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=True, speaker="Ethan")
    audio_np = audio.reshape(-1).detach().cpu().numpy()

    # 定义文件名
    filename_base = f"sample_{idx:04d}"

    # 保存文本
    txt_path = os.path.join(output_dir, f"{filename_base}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(ad_text)

    # 保存语音
    wav_path = os.path.join(output_dir, f"{filename_base}.wav")
    sf.write(wav_path, audio_np, samplerate=24000)

    print(f"Saved to {txt_path} and {wav_path}")
