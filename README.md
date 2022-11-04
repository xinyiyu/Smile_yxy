# Smile measurement and modeling
This folder contains code and raw data related to smile measurement and modeling

## Usage
1. Create a conda environment:

    ```
    conda create --name smile python=3.7
    conda activate smile
    pip install -r requirements.txt
    ```

2. Activate the conda environment:

    ```
    conda activate smile
    ```

3. Change the working directory (`Udemy_simle` or `5miles_smile`), e.g:
    
    ```
    cd Udemy_smile
    ```
    
## Udemy
Suppose the folder `videos/` contains all the videos to be processed. You can also split the folder into several subfolders if there are a lot videos to be processed. For example, there are 897 personal development course videos. I split them into `pd_1-100`, `pd_101-200`, ..., `pd_801-897`. This will facilitate multiprocessing.

1. Detect and recognize speaker's faces from video.
Please read the code `detect_faces.py` and specify required variables. For example, if the extraction frequency is 5 per second, the root `/home/<username>/Downloads` contains video folders `pd_1-100`, `pd_101-200`, ..., `pd_801-897`, the aligned face images will be saved under `pd/<pd_1-100, pd_101-200, ..., pd_801-897>/freq5_aligned/`, the frame information (whether face appears in the frame) and facial landmarks will be saved under `pd/<pd_1-100, pd_101-200, ..., pd_801-897>/freq5_scores/`. These 9 video folders will be processed in parallel. So you need to adjust the thread number for processing each folder so that the total number of threads does not exceed cpu core number. Finally, run the script:

    ```
    python code/detect_faces.py
    ```
    
2. Estimate emotions of speaker's faces.
Specify required variables in `estimate_emotion.py` similar to Step 1. The estimated emotions will be saved under `pd/<pd_1-100, pd_101-200, ..., pd_801-897>freq5_scores/`. Run the script:
    
    ```
    python code/estimate_emotion.py
    ```

3. Set `freq=1` and repeat Step 1 and Step 2. This is for filtering videos where the speakers do not show up but only their photos are displayed.

4. Estimate facial asymmetry, attractiveness and video quality by running `estimate_asymmetry.py`, `estimate_beauty.py` and `estimate_video_quality.py`, respectively.

5. Calculate independent variables related to smile for all videos. Please check `indep_variables.ipynb` (Python code). The result table containing smiling variables will be saved under `pd`.

6. Model the relationship between variables and marketing outcome. Please check `udemy_summary.ipynb` (Python code) and `udemy_model.ipynb` (R code).

## 5miles
* Detect faces: `5miles_smile/code/detect_faces.py`. `seller_pictures_info.csv` containing face numbers and face proximity will be saved under `scores/`.
* Estimate emotion: `5miles_smile/code/estimate_emotion.py`. `seller_emotion.csv` containing smiling, intensity and emotion probabilities will be saved under `scores/`.
* Predict facial attractiveness: `5miles_smile/code/estimate_beauty.py`. `seller_beauty.csv` containing facial attractiveness scores will be saved under `scores/`.
* Estimate facial asymmetry: `5miles_smile/code/estimate_asymmetry.py`. `seller_asymmetry.csv` containing facial asymmetry scores will be saved under `scores/`.
* Evaluate image quality: `5miles_smile/code/estimate_image_quality.py`. `seller_pictures_quality.csv` containing facial attractiveness scores will be saved under `scores/`.
(* Enter the folder `5miles_smile` and run programs, e.g., `python code/detect_faces.py`.)
