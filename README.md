## Context Encoders: Feature Learning by Inpainting

This is the Pytorch implement of [CVPR 2016 paper on Context Encoders](http://cs.berkeley.edu/~pathak/context_encoder/)

![corrupted](https://github.com/BoyuanJiang/context_encoder_pytorch/blob/master/val_cropped_samples.png)
![result](https://github.com/BoyuanJiang/context_encoder_pytorch/blob/master/val_recon_samples.png)
### 1) Semantic Inpainting Demo

1. Install PyTorch http://pytorch.org/

2. Clone the repository
  ```Shell
  git clone https://github.com/BoyuanJiang/context_encoder_pytorch.git
  ```
3. Demo

    Download pre-trained model on Paris Streetview from
    [Google Drive](https://drive.google.com/open?id=0B6oeoQaX0xmzS0RXXzNYZkZ3ZUk) OR [BaiduNetdisk](https://pan.baidu.com/s/1hsLzJPq)
    ```Shell
    cp netG_streetview.pth context_encoder_pytorch/model/
    cd context_encoder_pytorch/model/
    # Inpainting a batch iamges
    python test.py --netG model/netG_streetview.pth --dataroot dataset/val --batchSize 100
    # Inpainting one image 
    python test_one.py --netG model/netG_streetview.pth --test_image result/test/cropped/065_im.png
    ```

### 2) Train on your own dataset
1. Build dataset

    Put your images under dataset/train,all images should under subdirectory

    dataset/train/subdirectory1/some_images
    
    dataset/train/subdirectory2/some_images

    ...
    
    **Note**:For Google Policy,Paris StreetView Dataset is not public data,for research using please contact with [pathak22](https://github.com/pathak22).
    You can also use [The Paris Dataset](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) to train your model

2. Train
```Shell
python train.py --cuda --wtl2 0.999 --niter 200
```

3. Test

    This step is similar to [Semantic Inpainting Demo](#1-semantic-inpainting-demo)

    