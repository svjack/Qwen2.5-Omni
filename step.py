'''
sudo apt-get update && sudo apt-get install cbm ffmpeg git-lfs
git clone https://huggingface.co/datasets/svjack/Bakuman_Videos_Splited

#### 较好的方式是 使用 huggingface 已经部署的url 来做
'''

### pip install openai numpy soundfile "httpx[socks]"

import base64
import numpy as np
import soundfile as sf

from openai import OpenAI

client = OpenAI(
    api_key="sk-",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
            {"type": "text", "text": "使用中文描述视频内容。"}
        ],
    },
]


# Qwen-Omni only supports stream mode
completion = client.chat.completions.create(
    model="qwen-omni-turbo",
    messages=messages,
    modalities=["text", "audio"],
    audio={
        "voice": "Cherry", # Cherry, Ethan, Serena, Chelsie is available
        "format": "wav"
    },
    stream=True,
    stream_options={"include_usage": True}
)

text = []
audio_string = ""
for chunk in completion:
    if chunk.choices:
        if hasattr(chunk.choices[0].delta, "audio"):
            try:
                audio_string += chunk.choices[0].delta.audio["data"]
            except Exception as e:
                text.append(chunk.choices[0].delta.audio["transcript"])
    else:
        print(chunk.usage)

print("".join(text))
wav_bytes = base64.b64decode(audio_string)
wav_array = np.frombuffer(wav_bytes, dtype=np.int16)
sf.write("output.wav", wav_array, samplerate=24000)

### pip install openai numpy soundfile "httpx[socks]"

import base64
import numpy as np
import soundfile as sf

from openai import OpenAI

client = OpenAI(
    api_key="sk-",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
            {"type": "text", "text": "使用中文描述视频内容。"}
        ],
    },
]


# Qwen-Omni only supports stream mode
completion = client.chat.completions.create(
    model="qwen-omni-turbo",
    messages=messages,
    modalities=["text",],
    #audio={
    #    "voice": "Cherry", # Cherry, Ethan, Serena, Chelsie is available
    #    "format": "wav"
    #},
    stream=True,
    stream_options={"include_usage": True}
)

text = []
audio_string = ""
for chunk in completion:
    if chunk.choices:
        if True:
            try:
                audio_string += chunk.choices[0].delta.content
            except Exception as e:
                #text.append(chunk.choices[0].delta.audio["transcript"])
                pass
    else:
        print(chunk.usage)
print(audio_string)
