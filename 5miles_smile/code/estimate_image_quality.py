import os
import pandas as pd

if __name__ == '__main__':

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(f'{root}/IQA-BRISQUE/Python/libsvm/python/')
    cmd = f"python brisquequality.py {root}/seller_pictures {root}/scores/seller_pictures_quality.csv"
    os.system(cmd)
    
    quality = pd.read_csv(f"{root}/scores/seller_pictures_quality.csv", sep='\t')
    quality.rename(columns={'image': 'seller_ID'}, inplace=True)
    quality = quality.sort_values(by='seller_ID').reset_index(drop=True)
    quality.to_csv(f"{root}/scores/seller_pictures_quality.csv", sep='\t', index=False)
    
    