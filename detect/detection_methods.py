import torch
from PIL import Image
import os
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import json
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def init_detection(detect_method = "clip_itm", folder_path = './data/drawing/', json_path = './data/stereotype/stereotype.json', save_path = "./data/detection/"):
    '''
    detect_method: str ['clip_itm', 'clip_itc', 'blip_classification', 'clip_classification']
    folder_path: pictures dictionary
    json_path: path for stereotype json file
    save_path: dictionary for saving results
    '''
    results_dict = []
    with open(json_path) as annos:
        stereotype_json = json.load(annos)

    stereo_class_list = [dic["stereo_class"] for dic in stereotype_json]

    if detect_method == "clip_itm":
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
        detection = {}
        for class_id, stereo_class in enumerate(stereo_class_list):
            detection[stereo_class]={}
            stereo_pairs = stereotype_json[class_id]["stereo_pairs"]
            for pair in stereo_pairs:
                caption = pair["stereotype"]
                object = pair["object"]
                txt = text_processors["eval"](caption)
                folder_path = os.path.join("./data/drawing/", stereo_class, object+"_"+caption)
                images = os.listdir(folder_path)
                detection[stereo_class][object+"_"+caption] = []
                for image_name in images:
                    raw_image = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
                    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                    detection[stereo_class][object+"_"+caption].append({"img_name": image_name, "itm_score": itm_scores[:, 1].item()})
        results_dict.append(detection)
    elif detect_method == "clip_itc":
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
        detection = {}
        for class_id, stereo_class in enumerate(stereo_class_list):
            detection[stereo_class]={}
            stereo_pairs = stereotype_json[class_id]["stereo_pairs"]
            for pair in stereo_pairs:
                caption = pair["stereotype"]
                object = pair["object"]
                txt = text_processors["eval"](caption)
                folder_path = os.path.join("./data/drawing/", stereo_class, object+"_"+caption)
                images = os.listdir(folder_path)
                detection[stereo_class][object+"_"+caption] = []
                for image_name in images:
                    print(image_name)
                    raw_image = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    itc_score = model({"image": img, "text_input": txt}, match_head='itc')
                    detection[stereo_class][object+"_"+caption].append({"img_name": image_name, "itc_score": itc_score})
        results_dict.append(detection)
    elif detect_method == "clip_classification":
        model, vis_processors, txt_processors = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-B-16", is_eval=True, device=device)
        detection = {}
        for class_id, stereo_class in enumerate(stereo_class_list):
            detection[stereo_class]={}
            stereo_pairs = stereotype_json[class_id]["stereo_pairs"]
            for pair in stereo_pairs:
                caption = pair["stereotype"]
                object = pair["object"]
                txt = [txt_processors["eval"](""+caption), txt_processors["eval"]("not "+caption)]
                folder_path = os.path.join("./data/drawing/", stereo_class, object+"_"+caption)
                images = os.listdir(folder_path)
                detection[stereo_class][object+"_"+caption] = []
                for image_name in images:
                    print(image_name)
                    raw_image = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    sample = {"image": image, "text_input": txt}
                    clip_features = model.extract_features(sample)
                    image_features = clip_features.image_embeds_proj
                    text_features = clip_features.text_embeds_proj
                    sims = (image_features @ text_features.t())[0]
                    probs = torch.nn.Softmax(dim=0)(sims).tolist()
                    detection[stereo_class][object+"_"+caption].append({"img_name": image_name, "classification_score": probs})
        results_dict.append(detection)

    elif detect_method == "blip_classification":
        model, vis_processors, _ = load_model_and_preprocess("blip_feature_extractor", model_type="base", is_eval=True, device=device)
        detection = {}
        for class_id, stereo_class in enumerate(stereo_class_list):
            detection[stereo_class]={}
            stereo_pairs = stereotype_json[class_id]["stereo_pairs"]
            for pair in stereo_pairs:
                caption = pair["stereotype"]
                object = pair["object"]
                cls_names = ["" + caption, "Not "+caption]

                from lavis.processors.blip_processors import BlipCaptionProcessor
                text_processor = BlipCaptionProcessor(prompt=" ")

                cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]
                folder_path = os.path.join("./data/drawing/", stereo_class, object+"_"+caption)
                images = os.listdir(folder_path)
                detection[stereo_class][object+"_"+caption] = []
                for image_name in images:
                    print(image_name)
                    raw_image = Image.open(os.path.join(folder_path, image_name)).convert("RGB")
                    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    sample = {"image": image, "text_input": cls_names}
                    image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
                    text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0]
                    sims = (image_features @ text_features.t())[0] / model.temp
                    probs = torch.nn.Softmax(dim=0)(sims).tolist()
                    detection[stereo_class][object+"_"+caption].append({"img_name": image_name, "classification_score": probs})
        results_dict.append(detection)

    with open(save_path+detect_method+'.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)