import os
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import yaml

mtcnn = MTCNN(image_size=160, margin=0)

def preprocess_folder(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for img_name in tqdm(os.listdir(src_folder)):
        img_path = os.path.join(src_folder, img_name)
        img = Image.open(img_path)
        face = mtcnn(img)
        if face is not None:
            face = (face + 1) / 2
            face = Image.fromarray((face.permute(1,2,0).mul(255).byte().numpy()))
            face.save(os.path.join(dst_folder, img_name))

def preprocess(cfg_path="params.yaml"):
    with open(cfg_path) as f:
        params = yaml.safe_load(f)
        preprocess_folder(os.path.join(params['links']['end_dir'], params['data']['train_dir']), params['preprocessed_data']['train_dir'])
        preprocess_folder(os.path.join(params['links']['end_dir'], params['data']['valid_dir']), params['preprocessed_data']['valid_dir'])
        print("----PROCESSED----")

if __name__ == "__main__":
    preprocess()
