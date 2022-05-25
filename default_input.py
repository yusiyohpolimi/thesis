# input_json = '/multiverse/datasets/shared/DTLD/v2.0/v2.0/Berlin_test.json'
import json
import argparse

def main(args):
    """
    Given the input json file and the prediction outputs, generates json file 
    as in the DTLD format (like input file).
    Saves the dictionary as a json file to the indicated save path. 
    """
    with open(args.input) as file:
        parsed = json.load(file)
    images = parsed['images']
    
    for image in images: 
        del image["labels"]
 
    path = args.output
    with open(path, 'w+') as f:
        json.dump(parsed, f, indent=4) 
    print('Saved to', path)       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', 
        type=str, 
        default='/multiverse/datasets/shared/DTLD/v2.0/v2.0/Berlin_test.json', 
        help='input json file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='default_inp.json', 
        help='save path of the new json file'
    )  
    args = parser.parse_args()
    main(args)
