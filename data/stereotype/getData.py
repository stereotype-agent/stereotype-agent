import pandas as pd
from prompt.utils.json_process import load_json
import os

def get_IHC_data():
    df = pd.read_excel('data/toxic/IHC/stereotype.xlsx')
    return df

def get_IHC2_data():
    df = pd.read_excel('data/toxic/IHC/IHC2.xlsx')
    return df

def get_SBIC_data(dataset='SBIC_train'):
    df = pd.read_excel('data/toxic/SBIC/'+ dataset +'.xlsx')
    return df

def get_HateExplain_data():
    data_path = 'data/toxic/HateExplain/HateExplain.xlsx'
    if os.path.exists( data_path):
        df = pd.read_excel(data_path)
        return df
    data = load_json('data/toxic/HateExplain/HateExplain_dataset.json')
    sentences = []
    for i, key in enumerate(data):
        tokens = data[key]["post_tokens"]
        sentence = ' '.join(tokens)
        sentences.append([i+1, sentence])
    writer = pd.ExcelWriter(data_path)
    df = pd.DataFrame(sentences)
    df.columns = ["Unnamed: 0","Unnamed: 1"]
    df.to_excel(writer, index=False)
    writer.close()
    return df

def get_SMTD_data():
    df = pd.read_excel('data/toxic/SMTD/Social Media Toxicity Dataset.xls')
    return df

def get_DGHS_data():
    df = pd.read_excel('data/toxic/DGHS/DGHSv0.2.2.xlsx')
    return df

if __name__ == '__main__':
    df = get_DGHS_data()
    content = df["Unnamed: 1"]
    for sentence in content:
        print(sentence)
