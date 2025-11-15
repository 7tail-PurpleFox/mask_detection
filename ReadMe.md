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
- 下載kaggle資料集並解壓，將archive資料夾移至根目錄
> https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- 下載報告文獻資料集並解壓，將NewFace Mask Dataset資料夾移至根目錄
> https://data.mendeley.com/datasets/8pn3hg99t4/2
- 運行preprocess_mask_data.py生成mask_data資料夾
```
python preprocess_mask_data.py
```
- 如果你想看人臉辨識度低於min_conf的圖片，運行generate_zero_face_check_image.py
```
python generate_zero_face_check_image.py
```
- 如果你想看單張圖的人臉資訊，運行check_one_image_conf.py(記得改path)
```
python check_one_image_conf.py
```