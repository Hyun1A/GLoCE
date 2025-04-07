import os
import argparse
from dotenv import load_dotenv
from skimage import io
from pprint import pprint

import sys
sys.path.append('metrics/giphy')

from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.utils import evenly_spaced_sampling
from model_training.preprocessors.face_detection.face_detector import FaceDetector
from tqdm import tqdm
import pandas as pd
import re


def process_image(path):
    image = io.imread(path)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(image, image_size) for image, _ in face_images]
    return face_recognizer.perform(face_images)


def extract_celebrity_name(text):
    text = text.replace('-', ' ').replace('_0.jpg', '.jpg').replace('_0.png', '.jpg')
    # evaluation patterns
    patterns = [
        r"A portrait of (.*)_(\d+)\.jpg",
        r"An image capturing (.*) at a public event_(\d+)\.jpg",
        r"An oil painting of (.*)_(\d+)\.jpg",
        r"A sketch of (.*)_(\d+)\.jpg",
        r"o_(.*) in an official photo_(\d+)\.jpg",
        r"o_(.*)_(\d+)\.jpg",
        r"(.*) in an official photo_(\d+)\.jpg",
        r"(.*)_(\d+)\.jpg"        
    ]
    
    no_match = True

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:  
            return match.group(1)  
        
    if no_match:
        print(text)
        raise ValueError("The input image name does not match any of the expected patterns.")


if __name__ == '__main__':
    load_dotenv('metrics/giphy/examples/.env')
    parser = argparse.ArgumentParser(description='Inference script for Giphy Celebrity Classifier model')

    parser.add_argument('--image_folder', type=str, help='path or link to the image folder')
    parser.add_argument('--save_excel_path', type=str, help='path to save the excel file')


    args = parser.parse_args()


    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    gif_frames = int(os.getenv('APP_GIF_FRAMES', 20))

    # model_labels = Labels(resources_path=os.getenv('APP_DATA_DIR'))
    model_labels = Labels(resources_path="metrics/giphy/examples/resources")
    celeb_list = pd.read_csv('generation_prompts/celeb_data_for_pair.csv')

    face_detector = FaceDetector(
        "metrics/giphy/examples/resources",
        margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
        use_cuda=os.getenv('APP_USE_CUDA') == "true"
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path="metrics/giphy/examples/resources",
        use_cuda=os.getenv('APP_USE_CUDA') == "true",
        top_n=5 
    )

    image_files=os.listdir(args.image_folder)
    image_names=sorted(image_files)   #sort image files
    
    predictions_list=[]
    p_celebrity_list=[]  
    n_no_faces=0
    
    image_names = [val for val in image_names if 'jpg' in val or 'png' in val]
    newCSV = pd.DataFrame(columns=['filename', 'prompt', 'evaluation_seed', 'find_t', 'find_p'])
    
    cnt_fail_erase = 0
    cnt_success_p = 0
    
    fail_list= []
    
    for file in tqdm(image_names):
        print(file)
        image_path=os.path.join(args.image_folder,file)        
        predictions = process_image(image_path) # precdictions contain the probabilities of the top n celebrities for one image
        
        find_t = False
        find_p = False
        
        names = extract_celebrity_name(file).split(' and ')
        print(names, predictions)
        for name in names:
            try:
                if celeb_list[celeb_list['name'] == name].iloc[0]['train'] == 1:
                    target_celeb = name
                else:
                    pres_celeb = name
            except:
                breakpoint()

        for pred in predictions:
            pred_name = str(pred[0][0][0]).split('_[')[0].replace('_', ' ')
            if find_t == False and pred_name == target_celeb: 
                fail_list.append((file, pred[0][0][1]))
                find_t = True
                cnt_fail_erase += 1
            elif find_p == False and pred_name == pres_celeb and pred[0][0][1] > 0.9:
                find_p = True
                cnt_success_p += 1
                
        prompt = file.split('_')[1].replace('-', ' ')    
        newCSV.loc[len(newCSV)] = {'filename': file,
                                    'prompt': prompt, 
                                    'evaluation_seed': int(file.split('_')[-2]),
                                    'find_t': find_t,
                                    'find_p': find_p}
        
        
    print("erase_success: " ,len(image_names) - cnt_fail_erase)
    print("preserve_success: " , cnt_success_p)
    os.makedirs(args.save_excel_path, exist_ok=True)
    
    newCSV.to_csv(f'{args.save_excel_path}/success_prompts.csv')
    