# %%
import albumentations as A
import argparse, random
import os, shutil, cv2, glob
import time
import matplotlib.pyplot as plt
from typing import List
from multiprocessing.pool import Pool
from collections import defaultdict

def clamp(x):
    return min(max(0.0, x), 1.0)

# %%
# write augmented images and annotations
def write_augmented_images_bboxes(i, old_image_name, image_ext, timage, bboxes, parent_folder, IMGS, LABELS, AUGMENTED_DATASET):
    # print(f"\tInside write_aug")
    image_basename = 'aug_' + old_image_name + '_' + str(i) + '.' + image_ext
    label_basename = 'aug_' + old_image_name + '_' + str(i) + '.txt'
    aug_image_path = os.path.join(parent_folder, AUGMENTED_DATASET, IMGS, image_basename)
    aug_label_path = os.path.join(parent_folder, AUGMENTED_DATASET, LABELS, label_basename)

    # print(aug_image_path)
    # print(aug_label_path)
    cv2.imwrite(aug_image_path, timage)

    f = open(aug_label_path, 'w')
    with open(aug_label_path, 'w') as f:
        lines = []
        for bbox in bboxes:
            x, y, w, h, cat_idx = map(str, bbox)
            l = [cat_idx, x, y, w, h]
            line = ' '.join(l) + '\n'
            lines.append(line)
        f.writelines(lines)

# %%
def show_image_annotation(image, bboxes):
    # change palette based on number of classes
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    dh, dw, _ = image.shape
    
    for dt in bboxes:
        x, y, w, h, cat_idx = dt
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(image, (l, t), (r, b), palette[cat_idx], 2)
    return image

# %%
def show(image, bboxes, t_image, t_bboxes):
    show_image_annotation(image, bboxes)
    show_image_annotation(t_image, t_bboxes)

    plt.figure(figsize=(16,16))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title('Original image')

    plt.subplot(1,2,2)
    plt.imshow(t_image)
    plt.title('Augmented image')

    plt.show()

# %%
def bounding_box_class_labels(image_path, label_path) -> List[float]:
    img = cv2.imread(image_path)

    fl = open(label_path, 'r')
    data = fl.readlines()
    fl.close()

    bounding_boxes = []

    for dt in data:
        cat_idx, x, y, w, h = map(float, dt.split(' '))
        x, y, w, h = map(clamp, [x, y, w, h])
        bounding_boxes.append([x,y,w,h, int(cat_idx)])    
    return bounding_boxes

# %%
# augmentation super list
aug_super_list = [    
    A.augmentations.transforms.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), always_apply=False, p=0.3), # yes
    A.augmentations.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3), # yes
    A.augmentations.transforms.Downscale (scale_min=0.30, scale_max=0.80, interpolation=None, always_apply=False, p=0.3), # yes
    A.augmentations.transforms.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.3), # yes
    A.augmentations.transforms.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.3), # yes
    A.augmentations.transforms.FancyPCA (alpha=0.1, always_apply=False, p=0.3), # yes
    # A.augmentations.transforms.GaussianBlur (blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.3), # yes
    A.augmentations.transforms.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.3), #yes
    # A.augmentations.transforms.GlassBlur (sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5), #no
    A.augmentations.transforms.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.2), #yes
    A.augmentations.transforms.MedianBlur (blur_limit=3, always_apply=False, p=0.3), #yes
    # A.augmentations.transforms.MotionBlur (blur_limit=7, always_apply=False, p=1), #no, yes-vehicle
    A.augmentations.transforms.MultiplicativeNoise (multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False, p=0.3), #yes
    # A.augmentations.transforms.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0), # no
    A.augmentations.transforms.Posterize (num_bits=4, always_apply=False, p=0.3), #yes
    # A.augmentations.transforms.RandomBrightness (limit=0.2, always_apply=False, p=0.2), #yes
    A.augmentations.transforms.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.2), # yes
    A.augmentations.transforms.RandomContrast (limit=0.2, always_apply=False, p=0.3), # yes
    # A.augmentations.transforms.RandomFog (fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=True, p=1), # no
    A.augmentations.transforms.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.2), # yes
    # A.augmentations.transforms.RandomRain (slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.3), # yes-vehicle
    # A.augmentations.transforms.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.1), # yes
    # A.augmentations.transforms.RandomSnow (snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=True, p=1), # no
    # A.augmentations.transforms.RandomSunFlare (flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=True, p=1),# no
    A.augmentations.transforms.RandomToneCurve (scale=0.1, always_apply=False, p=0.3), # yes
    A.augmentations.transforms.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.3), #yes
    # A.augmentations.transforms.RingingOvershoot (blur_limit=(7, 15), cutoff=(0.7853981633974483, 1.5707963267948966), always_apply=True, p=1), # no
    # A.augmentations.transforms.Sharpen (alpha=(0.2, 0.5), lightness=(0.5, 1), always_apply=False, p=0.5), #yes=vehicle
    # A.augmentations.transforms.Solarize (threshold=128, always_apply=True, p=1), # no
    # A.augmentations.transforms.Superpixels (p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=True, p=1), # no
    # A.augmentations.transforms.ToFloat (max_value=None, always_apply=True, p=1.0), # no
    A.augmentations.transforms.UnsharpMask (blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=False, p=0.2), # yes
    A.augmentations.transforms.HorizontalFlip(p=0.5), # yes
    # A.augmentations.transforms.ToGray(p=0.3) #yes]
]


# %%
def single_image_augmentation(arg):
    i, image_name, image_ext, image_path, label_path, parent_folder, IMGS, LABELS, AUGMENTED_DATASET, show_flag = arg
    image = cv2.imread(image_path)

    bboxes = bounding_box_class_labels(image_path, label_path)    
   
    # defining an augmentation pipeline
    aug_dist = defaultdict(int)
    n = 3
    aug_pipeline = random.sample(aug_super_list, n)
    # print('-------------------------------------')
    # print(*aug_pipeline, sep='\n')
    for a in aug_pipeline:
        aug_dist[a] += 1

    transform = A.Compose(aug_pipeline, bbox_params=A.BboxParams(format='yolo', min_area=10, min_visibility=0.10))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    # print(f'NAME - {image_name}/{i}, IMAGE- {image.shape}, BBOXES = {bboxes}\nTRANSFORMED IMAGE- {transformed_image.shape}, T_BBOXES = {transformed_bboxes}\n')
    write_augmented_images_bboxes(i, image_name, image_ext, transformed_image, transformed_bboxes, parent_folder, IMGS, LABELS, AUGMENTED_DATASET)
    
    if show_flag:
        show(image, bboxes, transformed_image, transformed_bboxes)
    return aug_dist

# %%
def data_augmentation(k, img_source, show_flag, seq_flag):

    if img_source.endswith('.txt'):
        suf = os.path.basename(img_source[:-4])
        with open(img_source, 'r') as f:
            imagepaths = f.readlines()
        imagepaths = [imagepath[:-1] for imagepath in imagepaths]
        parent_folder = os.path.join(img_source, '..')
    else:
        suf = ''
        imagepaths = glob.glob(os.path.join(img_source, '*.*g*'))
        parent_folder = img_source[:-7]
    
    print(f'Parent folder: {parent_folder}')
    print(f'len of images: {len(imagepaths)}')

    IMGS = 'images'
    LABELS = 'labels'
    AUGMENTED_DATASET = f'AugData_{suf}'

    # make augmented folder
    aug_folder = os.path.join(parent_folder, AUGMENTED_DATASET)
    if os.path.exists(aug_folder):
        shutil.rmtree(aug_folder)
    print(f'Augmented folder: {aug_folder}')
    os.mkdir(aug_folder)
    for subdir in [IMGS, LABELS]:
        d = os.path.join(aug_folder, subdir)
        os.mkdir(d)
    
    arg_list = []
    for image_path in imagepaths:
        image_basename = os.path.basename(image_path)
        basename, image_ext = image_basename.split('.')
        label_basename = basename + '.txt'
        label_path = os.path.join(parent_folder, LABELS, label_basename)

        if not os.path.exists(label_path):
            continue
        
        for i in range(k):
            arg = (i+1, basename, image_ext, image_path, label_path, parent_folder, IMGS, LABELS, AUGMENTED_DATASET, show_flag)
            arg_list.append(arg)

    print(f'No of original images = {len(imagepaths)}\nNo of augmented images to be made = {len(arg_list)}')
    total_aug_dist = defaultdict(int)
    
    if seq_flag:
        print(f'Doing sequentially.')
        # list(map(single_image_augmentation, arg_list))
        for arg in arg_list:
           d = single_image_augmentation(arg)
           for k, v in d.items():
            total_aug_dist[k] += v
    else:
        print(f'Using multithreading.')
        with Pool() as pool:
            print(f'Pool loop {pool}')
            start_time_ = time.perf_counter()
            list(pool.imap_unordered(single_image_augmentation, arg_list))
            # for res in results:
            #     for k, v in res.items():
            #         total_aug_dist[k] += v
            print(f'It took {time.perf_counter() - start_time_:.2} after making the pool')
    # except Exception:
    #     print(f"ERROR found in augmenting -- {image_basename}")
    return total_aug_dist.values()

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_source', type=str, help='folder/*.txt containing images list')
    parser.add_argument('--label_source', type=str, default='', help='None if images, labels are in the same folder, folder/*.txt containing images list')
    parser.add_argument('--n_aug', type=int, default=2, help='max number of augmented images to obtain from an image')
    opt = parser.parse_args()

    show_flag = False # to show augmented images while being created
    seq_flag = False # to perform sequentially or using multithreading

    start_time = time.perf_counter()
    total_aug_dist = data_augmentation(opt.n_aug, opt.image_source, show_flag, seq_flag)
    end_time = time.perf_counter()
    print(total_aug_dist)
    print(f'Finished augmentation in {round(end_time - start_time, 2)} sec.')
