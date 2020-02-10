# SLD readme

## A quick start

### Environment

Due to the fact that CUDA/CUDNN version on my lab server is not compatible with the newest version of pytorch, I wrote this code based on Pytorch 0.4.0. Mostly it's the same as the newest version except for the `gumbel_softmax` method. So I suggest:

* Using pytorch 0.4.0 to run the code

or 

* Looking up for the newest API and change to the newest version.

I prefer to the later solution. My lab server is updated a few weeks before so I could also use the newest version of Pytorch now.

### Training

Training the SLD model consists of 3 steps:

* Pretraining the speaker-listener by running `pretrain.sh`.

  This is the same as the simplest version.

* Pretraining the discriminator by running `discriminator_training.sh`

  Make sure that *id*, *dis_id*, *checkpoint_path* match with that in pretrain so you can load the pretrained speaker-listener model properly, otherwise it might not be good to train a valid discriminator.

* Joint training by running `joint_train.sh`

  Similarly, *id*, *dis_id*, *checkpoint_path* should be set properly. I'm not 100% sure the latest scripts match each other but I think they do. Or if any problem occurs because this, check the code:

  `main.py`: search for `torch.save` to see where the checkpoints are saved.

  `models/__init__.py`: see where the checkpoints are loaded from.

### Evaluation

* Generation: run `eval.sh`. Make sure you're loading the best model checkpoint.

## Some implementation stuffs

*I can not remember all the details about the code but I'll try to write as concrete as possible.*

### Data preparation

The most tricky part will be data loading, I remember some preprocesses are needed and preprocessed file should be in a separate folder. I compressed the files to `data.zip` on the branch and I think it contains all the files needed to train the model. **If something like `can't open file xxx` occurs, decompress the zip file to the repo folder and set the input_json and input_label_h5 correctly**  . 

This might not help because I think it's a step when using the first version of the image representations. However I can't find where we loaded the new version of feature from. Run the code and if any files are missing, report here.

### Speaker-listener

Almost the same with Jing's version. 

### Discriminator Training

A simplest classifier. 

### Joint Training

Using gumbel-softmax to do the seqGAN training. Relevant codes are in `TransformerModel.py:56`. As is stressed before, this suffers from a version compatibility problem. Change this part.

### Comprehension Evaluation

I overwrote the comprehension code by mistake, but the main implementation thinking is to directly rewrite the `eval_comprehension.py` file. 

For the speaker_listener part, the image feature and sentence feature are returned at every running step with the variable:`img_feat` and `sent_feat`. It will be quite easy to fetch the vectors. After that, comparing the similarity with the golden label will result in the comprehension accuracy.

*However, when I was running the comprehension code, I can only get a very small accuracy. Maybe the hinge loss is not set properly. One way to check is to run the comprehension code with the pretrained speaker-listener model to see if it can get a reasonable result. If not, it might not be useful to use this kind of comprehension method for this model unless we can find the mistakes. I remember I double checked the hinge loss part and doesn't see any abnormal phenomenon.*

