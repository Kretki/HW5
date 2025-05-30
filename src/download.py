import os
import zipfile
import urllib.request
from tqdm import tqdm
import yaml

def download(url, dest):
    if os.path.exists(dest):
        return
    with urllib.request.urlopen(url) as resp, open(dest, 'wb') as out:
        total = int(resp.info().get('Content-Length').strip())
        with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest)) as pbar:
            while True:
                chunk = resp.read(1024*8)
                if not chunk: break
                out.write(chunk); pbar.update(len(chunk))

def extract_zip(src, dst):
    with zipfile.ZipFile(src, 'r') as z:
        z.extractall(dst)

def download_celeba(cfg_path="params.yaml"):
    with open(cfg_path) as f:
        params = yaml.safe_load(f)
        os.makedirs(params['links']['end_dir'], exist_ok=True)

        celeba_zip = os.path.join(params['links']['end_dir'], "celeba.zip")
        download(params['links']['celeba_url'], celeba_zip)
        extract_zip(celeba_zip, params['links']['end_dir'])
        print("----DOWNLOADED----")

if __name__ == "__main__":
    download_celeba()