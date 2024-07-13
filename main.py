import os,sys
sys.path.append(os.getcwd())
import os
import argparse
from prompt.getStereotypes import get_stereotypes
from detect import detection_methods


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = 'sk-xxxx'
    parser = argparse.ArgumentParser()
    parser.add_argument('--stereotypes_path', '-sp', type=str,
                        default='data/stereotype/processed/stereotype_test.json')
    parser.add_argument('--detect_method', '-dm', choices=['clip_itm', 'clip_itc', 'blip_classification', 'clip_classification'])
    parser.add_argument('--data', '-d', type=str, default="SBIC_test",
                        help='all: return all stereotypes stored in stereotypes_path; '
                             'IHC/IHC2: add stereotypes extracted from IHC dataset;'
                             'SBIC_train/SBIC_test/SBIC_dev: add stereotypes extracted from SBIC dataset;'
                             'HateExplain: add stereotypes extracted from HateExplain dataset;'
                             'SMTD: add stereotypes extracted from SMTD dataset;'
                             'DGHS: add stereotypes extracted from DGHS dataset;'
                             'or input a sentence for stereotype extraction and '
                             'add this stereotype to all stereotypes∂ stored in stereotypes_path')
    parser.add_argument('--stereo_model', '-sm', choices=['gpt','rule'], type=str, default="gpt",
                        help='rule-based method or gpt-based method for stereotype extraction')
    parser.add_argument('--prompt', '-p', default=None)
    parser.add_argument('--llm_model', '-lm', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--temperature', '-t', type=float, default=0.2)
    parser.add_argument('--max_tokens', '-mt', type=int, default=256)
    args = parser.parse_args()

    if args.data in ["IHC","IHC2","SBIC_train","SBIC_test","SBIC_dev","HateExplain","SMTD","DGHS"]:
        args.stereotypes_path = 'data/stereotype/processed/'+args.data+'.json'
        args.log_path = 'data/stereotype/' + args.data + '.log'
        f = open(args.log_path, 'a')
        sys.stdout = f
        sys.stderr = f
    # args.data = 'balabala' # or input a sentence
    stereotypes_json = get_stereotypes(args.stereotypes_path, args.data, args.stereo_model, args.prompt,
                                       args.llm_model, args.temperature, args.max_tokens)
    

    detection_methods.init_detection(detect_method=args.detect_method, json_path=args.stereotypes_path)