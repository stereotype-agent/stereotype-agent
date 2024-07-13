from prompt.utils.GPT_API import get_response, construct_msg
import pandas as pd
import time

# Define a function to find stereotypes
def gpt_find_stereotypes(stereotypes_json, text, data_source, prompt, model, temperature, max_tokens):
    if prompt == None:
        prompt = "Now you are a stereotype extractor, " \
                    "Please help me extract the stereotypes contained in the content I sent. " \
                    "If the content sent contains stereotypes, you need to complete the following three tasks: " \
                    "1. The first task is to determine the category stereotype_class to which the stereotype belongs. The categories to which stereotypes belong include the following three: (1) gender/sexuality, (2) race/ethnicity, (3) religion. If it does not belong to the above three, return None. " \
                    "2. The second task is to determine the object object contained in the stereotype. If the category stereotype_class to which the stereotype belongs is gender/sexuality, then determine which of the following genders the object contained in it belongs to: (1) male, (2) female. " \
                    "If the category of the stereotype is race/ethnicity, then determine which of the following regions or ethnic groups the object of its connotation belongs to: (1) African, (2) European, (3) Asian, (4) Latino, (5) indigenous, (6) white, (7) black, (8) middle eastern. " \
                    "If the category of the stereotype is religion, then determine which of the following religions the object of its connotation belongs to: (1) Christian, (2) Muslim, (3) Buddhist, (4) Hindu, (5) Sikh, (6) Catholic, (7) Jew. " \
                    "3. The third task needs to be divided into two steps. The first step is to extract the stereotype, and the second step is to add a modifier describing a person to the original stereotype, that is, \"person\", so that the stereotype becomes a description of a person's characteristics rather than a simple thing or characteristic. For example: suppose the stereotype you extracted is \"small eyes\", you need to rewrite it as \"people with small eyes\". " \
                    "If the content I sent does not contain stereotypes, it returns None. " \
                    "For example: The sentence I sent is 'Look at this black cotton picker.'," \
                    "You need to summarize a Json format data output:" \
                    "{\"stereotype_class\":\"race/ethnicity\",\"object\": \"black\",\"stereotype\":\"people who is a cotton picker\"}"
    current_retry = 1
    if isinstance(text, str):
        content = text
        messages = construct_msg(prompt, content)
        response = get_response(model, messages, temperature, max_tokens, current_retry)
        while response == 'error':
            time.sleep(10 * current_retry)
            response = get_response(model, messages, temperature, max_tokens, current_retry)
            current_retry += 1
        stereotypes_json = process_response(response, stereotypes_json, data_source)
    elif isinstance(text, pd.DataFrame):
        content = text["Unnamed: 1"]
        for sentence in content:
            messages = construct_msg(prompt, sentence)
            response = get_response(model, messages, temperature, max_tokens, current_retry)
            while response == 'error':
                time.sleep(10 * current_retry)
                response = get_response(model, messages, temperature, max_tokens, current_retry)
                current_retry += 1
            stereotypes_json = process_response(response, stereotypes_json, data_source)
    return stereotypes_json


def process_response(response, stereotypes_json, data_source):
    str_content = response['choices'][0]["message"]["content"]
    stereo_pairs = str_content.split("\n")

    for stereo_pair in stereo_pairs:
        try:
            stereo_pair = eval(stereo_pair)
        except SyntaxError as e:
            print(e, stereo_pair)
        except NameError as e:
            print(e, stereo_pair)

        try:
            if isinstance(stereo_pair, dict):
                if stereo_pair["object"] == None:
                    continue
                if stereo_pair["stereotype_class"] == "gender/sexuality":
                    pair_id_prex = "1"
                    class_key = "gender_sexuality"
                    if stereo_pair["object"] == "male":
                        pair_id_midx = "1"
                    elif  stereo_pair["object"] == "female":
                        pair_id_midx = "2"
                    else:
                        print("Object: ", stereo_pair["object"], " does not exist!", " Stereotype: ", stereo_pair["stereotype"])
                        continue
                elif stereo_pair["stereotype_class"] == "race/ethnicity":
                    pair_id_prex = "2"
                    class_key = "race_enthnicity"
                    if "african" in stereo_pair["object"]:
                        stereo_pair["object"] = "african"
                        pair_id_midx = "1"
                    elif "european" in stereo_pair["object"]:
                        stereo_pair["object"] = "european"
                        pair_id_midx = "2"
                    elif stereo_pair["object"] == "asian":
                        pair_id_midx = "3"
                    elif "latin" in stereo_pair["object"]:
                        stereo_pair["object"] = "latino"
                        pair_id_midx = "4"
                    elif "white" in stereo_pair["object"]:
                        stereo_pair["object"] = "white"
                        pair_id_midx = "5"
                    elif "negro" in stereo_pair["object"] or "black" in stereo_pair["object"] or "non-white" in stereo_pair["object"]:
                        stereo_pair["object"] = "black"
                        pair_id_midx = "6"
                    elif "arab" in stereo_pair["object"] or "eastern" in stereo_pair["object"]:
                        stereo_pair["object"] = "middle eastern"
                        pair_id_midx = "7"
                    else:
                        print("Object: ", stereo_pair["object"], " does not exist!", " Stereotype: ", stereo_pair["stereotype"])
                        continue
                elif stereo_pair["stereotype_class"] == "religion":
                    pair_id_prex = "3"
                    class_key = "religion"
                    if "christ" in stereo_pair["object"]:
                        stereo_pair["object"] = "christian"
                        pair_id_midx = "1"
                    elif "musli"  in stereo_pair["object"] or "islam" in stereo_pair["object"]:
                        pair_id_midx = "2"
                        stereo_pair["object"] = "muslim"
                    elif "buddh" in stereo_pair["object"]:
                        pair_id_midx = "3"
                        stereo_pair["object"] = "buddhist"
                    elif "hindu" in stereo_pair["object"]:
                        stereo_pair["object"] = "hindu"
                        pair_id_midx = "4"
                    elif "catholic" in stereo_pair["object"]:
                        pair_id_midx = "5"
                        stereo_pair["object"] = "catholic"
                    elif "jew" in stereo_pair["object"]:
                        pair_id_midx = "6"
                        stereo_pair["object"] = 'jew'
                    else:
                        print("Object ", stereo_pair["object"], " does not exist!", " Stereotype: ", stereo_pair["stereotype"])
                        continue
                else:
                    print("Stereotype class: ", stereo_pair["stereotype_class"]," does not exist!", " Object: ", stereo_pair["object"], " Stereotype: ", stereo_pair["stereotype"])
                    continue
                stereotypes_json[class_key]["pair_counts"] += 1
                stereotypes_json[class_key][stereo_pair["object"]] += 1
                processed_stereo_pair = {
                    "pair_id": pair_id_prex+'_'+str(stereotypes_json[class_key]["pair_counts"])+'_'+pair_id_midx+'_'+str(stereotypes_json[class_key][stereo_pair["object"]]),
                    "object": stereo_pair["object"],
                    "stereotype": stereo_pair["stereotype"],
                    "source": data_source,
                }
                stereotypes_json[class_key]["stereo_pairs"].append(processed_stereo_pair)
            elif isinstance(stereo_pair, tuple):
                for pair in stereo_pair:
                    if pair["object"] == None:
                        continue
                    if pair["stereotype_class"] == "gender/sexuality":
                        pair_id_prex = "1"
                        class_key = "gender_sexuality"
                        if pair["object"] == "male":
                            pair_id_midx = "1"
                        elif pair["object"] == "female":
                            pair_id_midx = "2"
                        else:
                            print("Object: ", pair["object"], " does not exist!", " Stereotype: ",
                                  pair["stereotype"])
                            continue
                    elif pair["stereotype_class"] == "race/ethnicity":
                        pair_id_prex = "2"
                        class_key = "race_enthnicity"
                        if "african" in pair["object"]:
                            pair["object"] = "african"
                            pair_id_midx = "1"
                        elif "european" in pair["object"]:
                            pair["object"] = "european"
                            pair_id_midx = "2"
                        elif pair["object"] == "asian":
                            pair_id_midx = "3"
                        elif "latin" in pair["object"]:
                            pair["object"] = "latino"
                            pair_id_midx = "4"
                        elif "white" in pair["object"]:
                            pair["object"] = "white"
                            pair_id_midx = "5"
                        elif "negro" in pair["object"] or "black" in pair["object"] or "non-white" in pair["object"]:
                            pair["object"] = "black"
                            pair_id_midx = "6"
                        elif "arab" in pair["object"] or "eastern" in pair["object"]:
                            pair["object"] = "middle eastern"
                            pair_id_midx = "7"
                        else:
                            print("Object: ", pair["object"], " does not exist!", " Stereotype: ",
                                  pair["stereotype"])
                            continue
                    elif pair["stereotype_class"] == "religion":
                        pair_id_prex = "3"
                        class_key = "religion"
                        if "christ" in pair["object"]:
                            pair["object"] = "christian"
                            pair_id_midx = "1"
                        elif "musli" in pair["object"] or "islam" in pair["object"]:
                            pair_id_midx = "2"
                            pair["object"] = "muslim"
                        elif "buddh" in pair["object"]:
                            pair_id_midx = "3"
                            pair["object"] = "buddhist"
                        elif "hindu" in pair["object"]:
                            pair["object"] = "hindu"
                            pair_id_midx = "4"
                        elif "catholic" in pair["object"]:
                            pair_id_midx = "5"
                            pair["object"] = "catholic"
                        elif "jew" in pair["object"]:
                            pair["object"] = 'jew'
                            pair_id_midx = "6"
                        else:
                            print("Object ", pair["object"], " does not exist!", " Stereotype: ", pair["stereotype"])
                            continue
                    else:
                        print("Stereotype class: ", pair["stereotype_class"], " does not exist!",
                              " Object: ", pair["object"], " Stereotype: ", pair["stereotype"])
                        continue
                    stereotypes_json[class_key]["pair_counts"] += 1
                    stereotypes_json[class_key][pair["object"]] += 1
                    processed_stereo_pair = {
                        "pair_id": pair_id_prex + '_' + str(stereotypes_json[class_key]["pair_counts"])+'_'+pair_id_midx+'_'+str(stereotypes_json[class_key][pair["object"]]),
                        "object": pair["object"],
                        "stereotype": pair["stereotype"],
                        "source": data_source,
                    }
                    stereotypes_json[class_key]["stereo_pairs"].append(processed_stereo_pair)
            else:
                print("TypeError! Pairs type: ", type(stereo_pairs), "pair type: ", type(stereo_pair))
        except KeyError as e:
            print(e, stereo_pair)
    return stereotypes_json
