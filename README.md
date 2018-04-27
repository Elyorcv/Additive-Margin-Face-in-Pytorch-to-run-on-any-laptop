# Additive-Margin-Face-in-Pytorch-to-run-on-any-laptop
### A simple face recognition system can run on any latop using Additive Margin Softmax trained model

The purpose of this project is to build a toy face recognition system that can run on my laptop.
There are several works I have done for this:
* using MSceleb dataset to train the AM-softmax model
    * the [original model is in tensorflow], now ported to pytorch.
    * got a 97% accuracy on lfw dataset without alignment
    * also update the [MTCNN pytorch] version a little bit, make it end to end pytorch
    * This project is based on pytorch0.3, but compatible with new pytorch0.4    


- Multiprocess in python to realize parallel processing of image capture and face detection
    - acceptable speed on my 3 years old computer
    - [here1] and [here2] is a demo video
    
[original model is in tensorflow]:https://github.com/Joker316701882/Additive-Margin-Softmax.git 
[MTCNN pytorch]:https://github.com/TropComplique/mtcnn-pytorch.git
[here1]:https://www.youtube.com/watch?v=JugNgxOXTXM&feature=youtu.be
[here2]:http://v.youku.com/v_show/id_XMzU2NzE3MDc3Ng==.html?spm=a2hzp.8244740.0.0

### How to use it

* 1 clone
    ```
    git clone https://github.com/TropComplique/mtcnn-pytorch.git
    ```

* 2 download the [weights] to model folder

* 3 to take a picture, run
    ```
    python take_pic.py
    ```
    press q to take a picture, it will only capture 1 highest possibility face if more than 1 person appear in the camera

* 4 or you can put any preexisting photo into the facebook directory, the file structure is following:
    
- facebook|
    - name1/photo.jpg
    - name2/photo.jpg
    - name3/photo.jpg
    - .....
    only 1 photo is allowed in each sub directory

- 5 to start
    ```
    python face_verify.py 
    ```

For training, please refer to msceleb_cosface_train.ipynb
[weights]:https://pan.baidu.com/s/1PwHjtGLAmAoG5LJkQk5LSQ
