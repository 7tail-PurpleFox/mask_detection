建立環境
===
首先要有python，我使用的是3.9.13版本
- 建立虛擬環境
```
python -m venv .venv
```
- 進入虛擬環境
```
.\.venv\Scripts\Activate.ps1 # Window
source .venv/bin/activate # Linux
```
- 下載package
```
pip install -r requirements.txt
```

---
下載資料集
===
- 下載kaggle資料集並解壓，將archive資料夾移至根目錄
> https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- 下載報告文獻資料集並解壓，將NewFace Mask Dataset資料夾移至根目錄
> https://data.mendeley.com/datasets/8pn3hg99t4/2
- 下載另一個kaggle資料集並解壓，將dataset資料夾改名成append並移至根目錄
> https://www.kaggle.com/datasets/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask

---
處理資料集
===
- 運行preprocess_mask_data.py生成mask_data資料夾
```
python preprocess_mask_data.py
```

---
preprocess_mask_data.py將生成mask_data資料夾，裡面的annotations資料夾存放每張圖片對應的xml檔，裡面有臉的座標資訊，images放圖片，clips放將根據xml的座標將臉切下來的圖片，zero_face_detected.txt寫了retina-face模型沒辨識到人臉的檔案路徑，如果你有將preprocess_mask_data.py裡的check變數改成True，還會生成check資料夾，裡頭是根據座標用顏色把臉框出來的圖片，可以拿來檢查。images和annotations可以用來訓練Yolo模型，clips可以拿來訓練Mobilenet模型。

---
preprocess_mask_data.py對每個資料集做不同處理，第一個Kaggle資料集每張圖片做4張增強，報告文獻資料集和第二個Kaggle資料集都增強過且沒有臉的座標，使用retina-face模型偵測出臉的座標後寫進xml檔，其中報告文獻資料集重複的人過多，因此只採用口罩沒戴好的資料夾，忽略Bandana資料夾(這不是口罩)，且每80張圖片採用一張(圖片數除80)。資料集處理完後，將臉的圖片切出來，由於模型切出來的座標大部分都很貼合臉，因此會先將長寬乘1.4倍再切，方便辨識口罩沒戴好的部分。

---
訓練模型(MobileNetV2)
===
- 運行MobileNet_train.py
```
python MobileNet_train.py
```
- 參數表，能自己加參數，預設的部分看code的第80行那裡
```
usage: MobileNet_train.py [-h] [--data_dir DATA_DIR] [--img_size IMG_SIZE IMG_SIZE] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--model_out MODEL_OUT]
                          [--plots_dir PLOTS_DIR]

Train MobileNetV2-based mask detector

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   dataset root directory
  --img_size IMG_SIZE IMG_SIZE
                        image size (h w)
  --batch_size BATCH_SIZE
                        batch size
  --epochs EPOCHS       number of epochs to train
  --model_out MODEL_OUT
                        output model filename
  --plots_dir PLOTS_DIR
                        directory to save training plots and reports
```
---
會生出h5檔的模型和training_plots資料夾，training_plots資料夾裡放了訓練時的圖表

---
使用模型偵測
===
- 運行Webcam.py
```
python Webcam.py
```
參數表，--confidence是模型準確率超過該數值才會顯示，預設的部分看code的第13行那裡
```
usage: Webcam.py [-h] [--confidence CONFIDENCE] [--model MODEL] [--size SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --confidence CONFIDENCE
  --model MODEL
  --size SIZE
```