import os
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import random

def augment_images(images_pd):
    augmented_pd = []
    target_pd = images_pd[images_pd['Cancer_Type'] != 'ductal_carcinoma']
    random.seed(2543673)

    for row in target_pd.itertuples():
        path = row.Path
        cancer_type = row.Cancer_Type
        im = Image.open(path)
        file_name = path.split("\\")[-1].replace(".png", "")
       
        im_flr = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((224, 224))
        im_ftb = im.transpose(Image.FLIP_TOP_BOTTOM).resize((224, 224))
        im_rotate_90  = im.transpose(Image.Transpose.ROTATE_90).resize((224, 224))
        im_rotate_180  = im.transpose(Image.Transpose.ROTATE_180).resize((224, 224))
        im_rotate_270  = im.transpose(Image.Transpose.ROTATE_270).resize((224, 224))
        im_transpose = im.transpose(Image.Transpose.TRANSPOSE).resize((224, 224))
        
        aug_imgs = [("flr", im_flr), ("ftb", im_ftb), ("90", im_rotate_90), ("180", im_rotate_180), ("270", im_rotate_270), ("transpose", im_transpose)]
        
        if cancer_type == "fibroadenoma":
            aug_imgs = random.sample(aug_imgs, 2)
        elif cancer_type == "lobular_carcinoma": 
            aug_imgs = random.sample(aug_imgs, 4)
        elif cancer_type == "mucinous_carcinoma":
            aug_imgs = random.sample(aug_imgs, 3)
        elif cancer_type == "papillary_carcinoma" or cancer_type == "tubular_adenoma":
            aug_imgs = random.sample(aug_imgs, 5)
        
        for  (type, img) in aug_imgs:
            img_path = f"source_data/{cancer_type}/augmentation/{file_name}_{type}.png"
            augmented_pd.append((img_path, cancer_type))
            img.save(img_path, optimize=True)
            
    augmented_pd = pd.DataFrame(augmented_pd, columns=["Path", "Cancer_Type"])
    images_pd = pd.concat([images_pd, augmented_pd], axis = 0, ignore_index=True)
    return images_pd
     
images_pd = pd.DataFrame(columns=["Path", "Cancer_Type"])
images_list = []
ohe = OneHotEncoder(sparse_output =False)

for (dirpath, dirnames, filenames) in os.walk("source_data"):
    if len(filenames) > 0:
        is_augmented = dirpath.split("\\")[2] == "augmentation"
        
        if is_augmented:
            continue
        
        cancer_type = dirpath.split("\\")[1]
        
        for file in filenames:
            images_list.append({"Cancer_Type" : cancer_type, "Path" : f"{dirpath}\{file}"})
            

images_pd = pd.DataFrame(images_list, columns=["Path", "Cancer_Type"])
print(images_pd.groupby('Cancer_Type').count())
images_pd = augment_images(images_pd)
print(images_pd.groupby('Cancer_Type').count())
cancer_type = images_pd[['Cancer_Type']]
ohe.fit(cancer_type)
ohe_features = ohe.get_feature_names_out()
encoding = pd.DataFrame(ohe.transform(cancer_type),columns=ohe_features)

X_train, X_test, y_train, y_test = train_test_split(images_pd, encoding, test_size=0.1, random_state=2543673, stratify=images_pd['Cancer_Type'])
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=2543673, stratify=X_train['Cancer_Type'])

X_train = X_train.drop(columns='Cancer_Type')
X_test = X_test.drop(columns='Cancer_Type')
X_valid = X_valid.drop(columns='Cancer_Type')


pd.concat([X_train, y_train], axis=1).to_csv("preprocessed_data/train.csv", index=False)
pd.concat([X_valid, y_valid], axis=1).to_csv("preprocessed_data/valid.csv", index=False)
pd.concat([X_test, y_test], axis=1).to_csv("preprocessed_data/test.csv", index=False)