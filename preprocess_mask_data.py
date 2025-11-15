import albumentations as A
import cv2
import os
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
import numpy as np
import tqdm

# 將兩個口罩資料集合併到同一個資料夾中
mask_path_1 = "archive/" # Kaggle資料集路徑，下載點: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
mask_path_2 = "NewFace Mask Dataset/" # 報告文獻資料集路徑，下載點: https://data.mendeley.com/datasets/8pn3hg99t4/2
target_path = "mask_data/" # 合併後的資料夾路徑
check = True # 是否產生檢查圖
out_img_dir = os.path.join(target_path, "images") # 輸出影像資料夾
out_xml_dir = os.path.join(target_path, "annotations") # 輸出XML資料夾
out_check_dir = os.path.join(target_path, "check") # 輸出檢查圖資料夾
clip_out_dir = os.path.join(target_path, "clips") # 輸出裁切人臉資料夾
zero_face_report = os.path.join(target_path, "zero_face_detected.txt") # 零人臉報告檔案
# confidence thresholds
min_conf = 0.4
# 每張圖要產生的增強版本數量
AUGMENTATIONS_PER_IMAGE = 4

# 建立目標資料夾
if not os.path.exists(target_path):
    os.makedirs(target_path)
else:
    print("目標資料夾已存在，請先刪除或更改目標資料夾名稱")
    exit()
    
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_xml_dir, exist_ok=True)
if check:
    os.makedirs(out_check_dir, exist_ok=True)
os.makedirs(clip_out_dir, exist_ok=True)
if os.path.exists(zero_face_report):
    os.remove(zero_face_report)

# 處理第一個資料集
print("Processing Dataset A...")
print(f"process {AUGMENTATIONS_PER_IMAGE} augmentations per image.")
# 資料夾設定
img_dir = os.path.join(mask_path_1, "images")
xml_dir = os.path.join(mask_path_1, "annotations")

face_be_eliminated = 0

# 增強組合
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2),
    A.HueSaturationValue(p=0.3),
    A.RandomGamma(p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], clip=True))

# 處理每張圖與XML
xml_files = sorted([f for f in os.listdir(xml_dir) if f.lower().endswith('.xml')])
for xml_file in tqdm.tqdm(xml_files, desc="Processing XML files", unit="file"):
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    img_file = root.find('filename').text
    img_path = os.path.join(img_dir, img_file)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # 取出 bboxes
    bboxes = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        ymin = int(xmlbox.find('ymin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymax = int(xmlbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
        
    # 將原始圖與 XML 複製到目標資料夾(xml要改成和增強一樣的格式)
    cv2.imwrite(os.path.join(out_img_dir, img_file), image)
    # 建立新的 XML (使用原始影像的檔名與尺寸)
    new_root = ET.Element("annotation")
    ET.SubElement(new_root, "filename").text = img_file
    size = ET.SubElement(new_root, "size")
    ET.SubElement(size, "width").text = str(image.shape[1])
    ET.SubElement(size, "height").text = str(image.shape[0])
    ET.SubElement(size, "depth").text = "3"
    for label, box in zip(labels, bboxes):
        if int(box[2])-int(box[0]) <=0 or int(box[3])-int(box[1]) <=0:
            face_be_eliminated += 1
            continue
        obj = ET.SubElement(new_root, "object")
        ET.SubElement(obj, "name").text = label
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(box[3]))
    new_tree = ET.ElementTree(new_root)
    new_tree.write(os.path.join(out_xml_dir, xml_file))

    # 產生多個增強版本
    base_name, ext = os.path.splitext(img_file)
    if ext == "":
        ext = ".png"

    for idx in range(AUGMENTATIONS_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, category_ids=labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['category_ids']

        

    

    # 每個增強版本也寫對應的 XML
    for idx in range(AUGMENTATIONS_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, category_ids=labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['category_ids']
        
        new_img_name = f"{base_name}_aug_{idx}{ext}"
        cv2.imwrite(os.path.join(out_img_dir, new_img_name), aug_img)
        # 在增強後的圖片上畫出 bboxes和標籤(測試用)
        if check:
            check_img = aug_img.copy()
            for box, label in zip(aug_bboxes, aug_labels):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(check_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
                cv2.putText(check_img, str(label), (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            check_img_path = os.path.join(out_check_dir, new_img_name)
            cv2.imwrite(check_img_path, check_img)

        xml_name = f"{os.path.splitext(xml_file)[0]}_aug_{idx}.xml"
        new_root = ET.Element("annotation")
        ET.SubElement(new_root, "filename").text = f"{base_name}_aug_{idx}{ext}"
        size = ET.SubElement(new_root, "size")
        ET.SubElement(size, "width").text = str(augmented['image'].shape[1])
        ET.SubElement(size, "height").text = str(augmented['image'].shape[0])
        ET.SubElement(size, "depth").text = "3"

        for label, box in zip(aug_labels, aug_bboxes):
            if int(box[2])-int(box[0]) <=0 or int(box[3])-int(box[1]) <=0:
                face_be_eliminated += 1
                continue
            obj = ET.SubElement(new_root, "object")
            ET.SubElement(obj, "name").text = label
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(box[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(box[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(box[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(box[3]))

        new_tree = ET.ElementTree(new_root)
        new_tree.write(os.path.join(out_xml_dir, xml_name))
print("Dataset A processing complete.")

# 處理第二個資料集
print("Processing Dataset B...")
print("use face detection to generate XML annotations.")
# input directories (adjust to your actual folders)
correct_img_dir = os.path.join(mask_path_2, "Correct")
incorrect_img_dir = os.path.join(mask_path_2, "Incorrect")

zero_face_count = 0
one_face_count = 0

correct_zero_face_count = 0
correct_one_face_count = 0

os.makedirs(out_xml_dir, exist_ok=True)
os.makedirs(out_img_dir, exist_ok=True)
if check:
    os.makedirs(out_check_dir, exist_ok=True)

prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"


# 下載模型相關檔案
if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/{prototxt}",
                prototxt)
    urlretrieve(f"https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/{caffemodel}",
                caffemodel)


net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# 支援的影像副檔名
IMG_EXTS = ('.jpg', '.jpeg', '.png')

def process_file(fname, root_dir, base_dir, conf_thresh):
    img_path = os.path.join(root_dir, fname)
    image = cv2.imread(img_path)
    if image is None:
        # 讀取失敗則跳過
        return

    h, w = image.shape[:2]

    # 使用 OpenCV DNN 偵測人臉
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # clip to image bounds
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        if endX <= startX or endY <= startY:
            continue
        faces.append({'box': (startX, startY, endX - startX, endY - startY), 'confidence': conf})

    if len(faces) == 0:
        with open(zero_face_report, 'a') as f:
            f.write(f"{img_path}\n")
        global zero_face_count
        zero_face_count += 1
        return
    else:
        global one_face_count
        one_face_count += 1
        
        if len(faces) > 1:
            # Only keep the highest confidence face
            faces = [max(faces, key=lambda x: x['confidence'])]
        
        # 複製圖片到目標資料夾
        cv2.imwrite(os.path.join(out_img_dir, fname), image)
        
        # 在 check 資料夾建立帶框的檢查圖並存檔
        if check:
            annot = image.copy()
            for f in faces:
                x, y, fw, fh = f.get('box', (0, 0, 0, 0))
                xmin = max(0, int(x))
                ymin = max(0, int(y))
                xmax = min(w, int(x + fw))
                ymax = min(h, int(y + fh))

                # 根據來源資料夾選擇框的顏色：正確配戴為綠色，錯誤為紅色
                color = (0, 255, 0) if base_dir == correct_img_dir else (0, 0, 255)
                cv2.rectangle(annot, (xmin, ymin), (xmax, ymax), color, 2)

                # 可選文字標註（與後續 XML 標籤一致）
                label = 'with_mask' if base_dir == correct_img_dir else 'mask_weared_incorrect'
                cv2.putText(annot, label, (xmin, max(ymin - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            out_check_path = os.path.join(out_check_dir, os.path.basename(img_path))
            cv2.imwrite(out_check_path, annot)
        

    ann_root = ET.Element('annotation')
    ET.SubElement(ann_root, 'filename').text = os.path.basename(img_path)
    size = ET.SubElement(ann_root, 'size')
    ET.SubElement(size, 'width').text = str(w)
    ET.SubElement(size, 'height').text = str(h)
    ET.SubElement(size, 'depth').text = '3'

    for f in faces:
        x, y, fw, fh = f.get('box', (0, 0, 0, 0))
        xmin = max(0, int(x))
        ymin = max(0, int(y))
        xmax = min(w, int(x + fw))
        ymax = min(h, int(y + fh))
        if xmax <= xmin or ymax <= ymin:
            face_be_eliminated += 1
            continue
        obj = ET.SubElement(ann_root, 'object')
        # 根據來源資料夾設定標籤：來自 correct_img_dir 的視為正確配戴
        label = 'with_mask' if base_dir == correct_img_dir else 'mask_weared_incorrect'
        ET.SubElement(obj, 'name').text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    xml_name = os.path.splitext(os.path.basename(img_path))[0] + '.xml'
    ET.ElementTree(ann_root).write(os.path.join(out_xml_dir, xml_name))
    
# 遍歷兩個資料夾，遞迴搜尋，但忽略名為 'Bandana' 的子資料夾
for base_dir in (correct_img_dir, incorrect_img_dir):
    if not os.path.isdir(base_dir):
        raise ValueError(f"Directory does not exist: {base_dir}")

    file_list = []
    for root_dir, dirs, files in os.walk(base_dir):
        # 忽略 Bandana 資料夾（避免進入）
        if 'Bandana' in dirs:
            dirs.remove('Bandana')

        for fname in files:
            if not fname.lower().endswith(IMG_EXTS):
                continue
            file_list.append((fname, root_dir))

    for fname, root_dir in tqdm.tqdm(file_list, desc=f"Processing {os.path.basename(base_dir)}", unit="file"):
        process_file(fname, root_dir, base_dir, min_conf)
    
    print(f"directory: {base_dir}, zero face count: {zero_face_count-correct_zero_face_count}, one face count: {one_face_count-correct_one_face_count}")
    if base_dir == correct_img_dir:
        correct_one_face_count = one_face_count
        correct_zero_face_count = zero_face_count       


print("Dataset B processing complete.")
print(f"Total image counts: {zero_face_count + one_face_count}")
print(f"Images with zero faces: {zero_face_count}")
print(f"Images with one face: {one_face_count}")     

# 產生clip
print("clipping faces from images...")
# 對應資料夾設定
with_mask = os.path.join(clip_out_dir, "with_mask")
without_mask = os.path.join(clip_out_dir, "without_mask")
mask_weared_incorrect = os.path.join(clip_out_dir, "mask_weared_incorrect")
os.makedirs(with_mask, exist_ok=True)
os.makedirs(without_mask, exist_ok=True)
os.makedirs(mask_weared_incorrect, exist_ok=True)
# 讀取XML檔案，並從對應圖片裁切人臉區域
xml_files = sorted([f for f in os.listdir(out_xml_dir) if f.lower().endswith('.xml')])
for xml_file in tqdm.tqdm(xml_files, desc="Clipping faces", unit="file"):
    tree = ET.parse(os.path.join(out_xml_dir, xml_file))
    root = tree.getroot()
    img_file = root.find('filename').text
    img_path = os.path.join(out_img_dir, img_file)
    image = cv2.imread(img_path)
    if image is None:
        continue
    face_count_in_image = 0
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        ymin = int(xmlbox.find('ymin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymax = int(xmlbox.find('ymax').text)
        face_clip = image[ymin:ymax, xmin:xmax]
        if name == 'with_mask':
            out_dir = with_mask
        elif name == 'without_mask':
            out_dir = without_mask
        else:
            out_dir = mask_weared_incorrect
        clip_name = f"{os.path.splitext(img_file)[0]}_{name}_{face_count_in_image}.jpg"
        try:
            cv2.imwrite(os.path.join(out_dir, clip_name), face_clip)
        except Exception as e:
            print(f"Error saving {clip_name}: {e}")
            print(f"name: {name}")
            print(f"Face box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}, image shape={image.shape}")
        face_count_in_image += 1
print("Clipping complete.")
print("All processing complete.")
print(f"Total xml files: {len(os.listdir(out_xml_dir))}")
print(f"Total image counts: {len(os.listdir(out_img_dir))}")
print(f"Total faces eliminated: {face_be_eliminated}")
print(f"Total clipped faces: {len(os.listdir(with_mask)) + len(os.listdir(without_mask)) + len(os.listdir(mask_weared_incorrect))}")
print(f"Total clipped faces with mask: {len(os.listdir(with_mask))}")
print(f"Total clipped faces without mask: {len(os.listdir(without_mask))}")
print(f"Total clipped faces with incorrect mask: {len(os.listdir(mask_weared_incorrect))}")