<p align="center">
  <img src="https://github.com/DavideStenner/Kaggle/blob/master/Google%20Landmark%20Retrieval%202020/google_landmark_2020_banner.png" />
</p>

In this challenge the task consists of creating an embedding for image retrieval purpose.

I create the embedding by extracting information from different block of efficientnet and for each block I add a convolution block a squeeze block and a GEM Pooling block.
The backbone is frozen to speed up convergence and the three block was trained (cnn, squeeze, gem) using semi-hard triplet loss.

There are two notebooks which use this approach:

- effb5-block-extractor.ipynb
- tpu-eff-b5-block-extractor.ipynb

To get final result, I run effb5-block-extractor which is run on gpu 4 times and tpu-eff-b5-block-extractor which uses tpu 1 times.
