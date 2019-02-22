import tensorflow as tf
import os

#用于指定第几块gpu
#使用nvidia-smi查看gpu使用情况

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
