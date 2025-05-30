import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from model import FaceNetModel
import yaml

def eval(cfg_path="params.yaml"):
    with open(cfg_path) as f:
        params = yaml.safe_load(f)

        model = FaceNetModel(classify=True, num_classes=params['train']['num_classes'])
        model.load_state_dict(torch.load("face_classifier.pt"))
        model.eval()

        prep = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        def get_embedding(img_path):
            img = Image.open(img_path).convert('RGB')
            x = prep(img).unsqueeze(0)
            with torch.no_grad():
                emb = model.encoder(x)
            return emb[0].cpu().numpy()


        def get_comparasion(folder_path):
            images = os.listdir(folder_path)
            names = {}
            for img in images:
                name = img.split("_")[0]
                if not name in names.keys():
                    names[name] = int(img.split("_")[1].split('.')[0])
                else:
                    names[name] = max(names[name], int(img.split("_")[1].split('.')[0]))
            embeddings = [get_embedding(os.path.join(folder_path, img)) for img in images]
            with open("Результаты.txt", "w") as f:
                for key in names.keys():
                    count = 0
                    for i in range(names[key]):
                        emb1 = embeddings[images.index(f'{key}_{i}.jpg')]
                        emb2 = embeddings[images.index(f'{key}_{i+1}.jpg')]
                        dist = np.linalg.norm(emb1 - emb2)
                        if dist > 1:
                            print(f"РАССТОЯНИЕ МЕЖДУ {key}_{i} И {key}_{i+1}: {dist:.3f}")
                            count += 1
                    if count:
                        print(f"СВОИ ЛИЦА {key} НЕ СОВПАДАЮТ {count} РАЗ", file=f)
                    else:
                        print(f"ВСЕ СВОИ ЛИЦА {key} НАХОДЯТСЯ ВНУТРИ СВОИХ КЛАССОВ", file=f)
                    count = 0
                    for other_key in names.keys():
                        if other_key != key:
                            emb2 = embeddings[images.index(f'{other_key}_0.jpg')]
                            dist = np.linalg.norm(emb1 - emb2)
                            if dist < 1:
                                print(f"РАССТОЯНИЕ МЕЖДУ {key}_0 И {other_key}_0: {dist:.3f}", file=f)
                                count += 1
                    if count:
                        print(f"ЧУЖИЕ ЛИЦА {key} И {other_key} СОВПАДАЮТ {count} РАЗ", file=f)
                    else:
                        print(f"ВСЕ ЧУЖИЕ ЛИЦА {key} И {other_key} ОТДЕЛИМЫ ПО КЛАССАМ", file=f)
        get_comparasion(params['preprocessed_data']['valid_dir'])


if __name__ == "__main__":
    eval()