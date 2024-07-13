import yaml
import json

from venus_api_base.venus_openapi import PyVenusOpenApi

with open('task.yaml', 'r') as file:
    task = yaml.safe_load(file)

model_id = {
    'ToonYou': 1000366,
    'AbsoluteReality': 1000362,
    'NovelAI': 9,
    'SD v1_4': 10, 
    'Anything': 11,
    'ChilloutMix': 15,
    'DeepFloyd IF': 1000162,
    'Any Pastel': 14,
    'AllinOne PixelModel': 13
}

model_name = ['ToonYou', 'AbsoluteReality', 'NovelAI', 'SD v1_4', 'Anything', 'ChilloutMix', 'DeepFloyd IF', 'Any Pastel', 'AllinOne PixelModel']

# 每次画图需要修改的参数
## 生成 prompt
text = '印第安人'
## 使用第几个模型，范围 0-8，共 9 个免费模型
model_idx = 0

generate_num = 5
venus_private_ak = task['venus_private_ak']
venus_private_sk = task['venus_private_sk']
api = PyVenusOpenApi(venus_private_ak, venus_private_sk)
# post 接口测试，batch_draw
data = {
    "app_group_id": 1, #替换为自己的应用组ID
    "creator": task['creator'],
    "notify_users": [task['creator']],
    "notify_slice_num": 4,
    "callback_api": "",
    "auth_token": "",
    # "caption_list": [{"sub_id": 1, "caption": text}, {"sub_id": 2, "caption": text}, {"sub_id": 3, "caption": text}, {"sub_id": 4, "caption": text}, {"sub_id": 4, "caption": text}],
    "caption_list": [{"sub_id": i, "caption": text} for i in range(generate_num)],
    "neg_prompt": "",
    "batch_size": 4,
    "model_id": model_id[model_name[model_idx]],
    "size_id": 7,
    "scale": 11,
    "step_num": 28,
    "sampler_id": 4,
    "task_type": "txt2img",
    "hires_fix": None,
    "controlnet_units": [],
    "style_model_prompts": [],
    # 面部修复相关参数, 不需要可以整个配置去掉
    "restore_faces": {
      "enabled": False,           # 是否开启
      "gfpgan_visibility": 0,     # 面部修复程度[0,1]
      "codeformer_visibility": 0, # 面部重建程度[0,1]
      "codeformer_weight": 0,     # 面部重建权重[0,1]
    },
    # 高分辨率相关参数, 不需要可以整个配置去掉
    "upscale": {
      "enabled": False,           # 是否开启
      "upscaling_crop": False,    # 是否裁剪以适应宽高比
      "upscaling_resize_w": 512,  # 裁剪后宽度
      "upscaling_resize_h": 512,  # 裁剪后高度
      "upscaler_1": "None",       # 高清算法1: None/Lanczos/Nearest/ESRGAN_4x/LDSR/R-ESRGAN 4x+/R-ESRGAN 4x+ Anime6B/ScuNET/ScuNET PSNR/SwinIR_4x
      "upscaler_2": "None",       # 高清算法2: None/Lanczos/Nearest/ESRGAN_4x/LDSR/R-ESRGAN 4x+/R-ESRGAN 4x+ Anime6B/ScuNET/ScuNET PSNR/SwinIR_4x
      "extras_upscaler_2_visibility": 0,  # 算法2可见度
    },
}

# 必须要这个header才行
header = {
    'Content-Type': 'application/json' 
}

print(json.dumps(data))
ret = api.post("http://v2.open.venus.oa.com/aidraw/api/batch_draw", header, json.dumps(data))
print(ret)
