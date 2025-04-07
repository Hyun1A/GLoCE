import os
from PIL import Image
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from argparse import ArgumentParser
import torch


@torch.no_grad()
def mean_clip_score(image_dir, prompts_path):
    device = torch.device("cpu") 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_df=pd.read_csv(prompts_path)
    texts=list(text_df['prompt'])
    image_filenames=os.listdir(image_dir)
    
    sorted_image_filenames = image_filenames
    similarities=[]
    for i in tqdm(range(len(sorted_image_filenames))):
        try:
            imagename=sorted_image_filenames[i].replace('\^J', '')
            textList = [val for val in texts if val.replace(' \n', '').replace('\n', '').replace(' ', '-') in imagename]
            textList.extend([val for val in texts if val.replace(' \n', '').replace('\n', '').replace(' ', '_') in imagename])
            
            if len(textList) == 0:
                continue
            text=textList[0]
            try:
                image=Image.open(os.path.join(image_dir,sorted_image_filenames[i]))
            except Exception as e:
                print(e)
            inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
            outputs = model(**{k : v.to(device) for k, v in inputs.items()})
            clip_score= outputs.logits_per_image[0][0].detach().cpu()  # this is the image-text similarity score
            similarities.append(float(clip_score))
        except Exception as e:
            print(e)
    similarities=np.array(similarities)
    
    
    mean_similarity=np.mean(similarities)
    std_similarity = np.std(similarities)

    print('-------------------------------------------------')
    print('\n')
    print(f"Mean CLIP score ± Standard Deviation: {mean_similarity:.4f}±{std_similarity:.4f}") 
    
    return mean_similarity 
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='path/to/generated_images')
    parser.add_argument("--prompts_path", type=str, default='./prompts_csv/coco_30k.csv')
    args = parser.parse_args()

    image_dir=args.image_dir
    prompts_path=args.prompts_path
    
    mean_clip_score(image_dir, prompts_path)