import os
import argparse
#import moviepy.editor as mov_editor
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
        r"(.*) in an official photo_(\d+)\.jpg"
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
    parser.add_argument('--image_folder', type=str, help='path or link to the image folder', default='path/to/generated_images')
    parser.add_argument('--save_excel_path', type=str, help='path to save the excel file', default='path/to/result_save_path')

    args = parser.parse_args()

    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    gif_frames = int(os.getenv('APP_GIF_FRAMES', 20))
    app_data_dir = "metrics/giphy/examples/resources"
    
    model_labels = Labels(resources_path=app_data_dir)

    face_detector = FaceDetector(
        app_data_dir,
        margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
        use_cuda=True
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=app_data_dir,
        use_cuda=True,
        top_n=5 
    )

    image_files=os.listdir(args.image_folder)
    image_names=sorted(image_files)   #sort image files
    
    predictions_list=[]
    p_celebrity_list=[]  
    n_no_faces=0
    
    image_names = [val for val in image_names if 'jpg' in val or 'png' in val]
    
    for file in tqdm(image_names):
        image_path=os.path.join(args.image_folder,file)
        
        predictions = process_image(image_path) # precdictions contain the probabilities of the top n celebrities for one image
        if len(predictions)==0:     # if no face detected
            n_no_faces+=1
            p_celebrity_list.append('N')  # give empty string if no face detected
            predictions_list.append([])
        else:
            predictions_new_label=[]
            for prediction in predictions[0][0]:
                celebrity_label, prob=prediction
                celebrity_label=str(celebrity_label)  
                # Modify label format
                celebrity_name=celebrity_label.split('_[',1)[0].replace('_',' ')
                prediction=(celebrity_name,prob)
                predictions_new_label.append(prediction)
            predictions_list.append(predictions_new_label)

            print('************************')
            print(predictions_new_label[0][0])
            #print(extract_celebrity_name(file))
            if predictions_new_label[0][0].lower() == extract_celebrity_name(file).lower():   #if the top1 prediction is correct
                p_celebrity_list.append(predictions_new_label[0][1])
            else:
                p_celebrity_list.append(0)   #if the top1 prediction is wrong, just give zero score
    print('-------------------')
    print('Total number of images with no faces detected:', n_no_faces)           

    # save as excel file
    df=pd.DataFrame(predictions_list, columns=['top1','top2','top3','top4','top5'])
    df.index=image_names
    df['p_celebrity_correct']=p_celebrity_list
    print('-------------------')
    print('Given face detected, the celebrity classification accuracy is:')
    
    # Calculate the number of non-zero and non-N values in p_celebrity_list and then divided by the number of non-N values.
    print(sum([1 for p in p_celebrity_list if p != 0 and p != 'N']) / sum([1 for p in p_celebrity_list if p != 'N']))

    if args.save_excel_path is not None:
        df.to_csv(os.path.join(args.save_excel_path, 'result.csv') )