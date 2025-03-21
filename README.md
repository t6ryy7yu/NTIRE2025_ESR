# [NTIRE 2025 Challenge on Efficient Super-Resolution - Efficient channel attention super-resolution network acting on space] @ [CVPR 2025]


## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## The Validation and Test datasets
You can organize validation and test dataset as follows:

```
|NTIRE2025_ESR_Challenge/
|--DIV2K_LSDIR_valid_HR/
|    |--000001.png
|    |--000002.png
|    |--...
|    |--000100.png
|    |--0801.png
|    |--0802.png
|    |--...
|    |--0900.png
|--DIV2K_LSDIR_valid_LR/
|    |--000001x4.png
|    |--000002x4.png
|    |--...
|    |--000100x4.png
|    |--0801x4.png
|    |--0802x4.png
|    |--...
|    |--0900.png
|--NTIRE2025_ESR/
|    |--...
|    |--test_demo.py
|    |--...
|--results/
|--......
```

## How to test our model?

1. `git clone https://github.com/t6ryy7yu/NTIRE2025_ESR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 52
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
    - or if you can run the code directly as follows:
     ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py
    ```


## Organizers
- Zongang Gao (gaozongang@qq.com)

If you have any question, feel free to contact me please.


## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 