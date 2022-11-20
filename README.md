# softline-assignment
~~~
+-- softline-assignment

|   +-- inference.py //script to inference on images from /inference/images
|   +-- model_training.py //to train the model
|   +-- my_utils.py //utils
|   +-- train.py //training loop

|   +-- Dataset
|   |   +-- train
|   |   |   +-- george
|   |   |   |   +-- george_images
|   |   |   +-- no_george
|   |   |   |   +-- no_george_images
|   |   +-- val
|   |   |   +-- george
|   |   |   |   +-- george_images
|   |   |   +-- no_george
|   |   |   |   +-- no_george_images

|   +-- inference
|   |   +-- images
|   |   |   +-- images.jpg(for inference)
|   |   +-- output.csv //result of inference

|   +-- paths
|   |   +-- testpaths.txt
|   |   +-- trainpaths.txt

|   +-- runs
|   |   +-- train
|   |   |   +-- best.pt       //best weights from training
|   |   |   +-- checkpoint.pt //model checkpoint weights
|   |   |   +-- resnet_finetuned.pt //my weights
~~~
