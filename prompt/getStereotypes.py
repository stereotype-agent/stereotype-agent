from data.stereotype.getData import get_IHC_data, get_IHC2_data, get_SBIC_data, get_HateExplain_data, get_SMTD_data, get_DGHS_data
from prompt.stereotypes.gpt_based import gpt_find_stereotypes
import os
from prompt.utils.json_process import save_json, load_json

def get_stereotypes(stereotypes_path, data, stereo_model, prompt=None, llm_model=None, temperature=None, max_tokens=None):
    if os.path.exists(stereotypes_path):
        stereotypes_json = load_json(stereotypes_path)
        if data == 'all':
            return stereotypes_json
    else:
        stereotypes_json = {
                "gender_sexuality": {
                    "pair_counts": 0,
                    "male": 0,
                    "female": 0,
                    "stereo_pairs": []
                },
                "race_enthnicity": {
                    "pair_counts": 0,
                    "african": 0,
                    "european": 0,
                    "asian": 0,
                    "latino": 0,
                    "white": 0,
                    "black": 0,
                    "middle eastern": 0,
                    "stereo_pairs": []
                },
                "religion": {
                    "pair_counts": 0,
                    "christian": 0,
                    "muslim": 0,
                    "buddhist": 0,
                    "hindu": 0,
                    "catholic": 0,
                    "jew": 0,
                    "stereo_pairs": []
                },
            }

    data_source = data
    if data == "IHC":
        content = get_IHC_data()
    elif data == "IHC2":
        content = get_IHC2_data()
    elif 'SBIC' in data:
        content = get_SBIC_data(data)
    elif data == "HateExplain":
        content = get_HateExplain_data()
    elif data == "SMTD":
        content = get_SMTD_data()
    elif data == "DGHS":
        content = get_DGHS_data()
    else:
        content = data
        data_source = "manually add"

    if stereo_model=="gpt":
        stereotypes_json = gpt_find_stereotypes(stereotypes_json, content, data_source, prompt, llm_model, temperature, max_tokens)

    if stereotypes_path!='':
        save_json(stereotypes_path, stereotypes_json)
    return stereotypes_json