# Lab 0: Run Vision Model & SLM on GPU
Before this lab begins, we strongly recommend you to watch the following videos to learn about basic machine learning knowledge and how to use PyTorch.

## **Prerequisite**

Basic Machine Learning Knowledge

- [Machine Learning 1](https://www.youtube.com/watch?v=Ye018rCVvOo)
- [Machine Learning 2](https://www.youtube.com/watch?v=bHcJCp2Fyxs)
- [Deep Learning](https://www.youtube.com/watch?v=Dr-WRlEFefw)
- [Back Propogation](https://www.youtube.com/watch?v=ibJpTrp5mcE)
- [Convolution](https://www.youtube.com/watch?v=OP5HcXJg2Aw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=9)

PyTorch Tutorial


- [Tensor Basics](https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=2)
- [PyTorch Example](https://www.youtube.com/watch?v=Jy4wM2X21u0&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=3)
- [Save & Load Model](https://www.youtube.com/watch?v=g6kQl_EFn84&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=7)
- [Torch Compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Export Model in ONNX Format](https://pytorch.org/docs/stable/onnx_torchscript.html)

<span id="Language Model Tutorial"></span>
Language Model & Transformer Tutorial

- [Introduction to Transformer](https://profuse-mule-ca0.notion.site/Transformer-Learning-Materials-3307acd25e7b4328bdf05d86afac27c7)
- [Huggingface](https://huggingface.co/learn/nlp-course/chapter1/1)
- [How LLM Works](https://www.youtube.com/watch?v=wjZofJX0v4M)

<span id="colab-tutorial"></span>
Colab Tutorial

- [Google Colab Tutoral by HUNG-YI LEE](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/Colab%20Tutorial%202022.pdf)

Appendix

- [PyTorch Official Tutorial Playlist](https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
- [Create Custom Dataset for images](https://www.youtube.com/watch?v=ZoZHd0Zm3RY&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=9)
- [CNN Example](https://www.youtube.com/watch?v=wnK3uWv_WkU&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=4)
- [PyTorch 2.0](https://youtu.be/GYQTJnD-yjQ?si=Oeg6xPsjpXqpkl7V)


## **Introduction**



- In the part 1 of this assignment, you'll be using the *MobileNetV2* architecture to build an image classifier for the *CIFAR-10* dataset. For part 2, you will interact with *Llama-3.2-1B-Instruct*, a chat model, and explore techniques to speed up its response time. Through this hands-on experience, you will learn about basic PyTorch and also how to run a language model.

- Please download the code we provided and follow the hints to finish the code.
**Code Link**: [Link](https://drive.google.com/file/d/1XATUOLU3PXX7lP0Y3R50xe0-4bjUSj-4/view?usp=sharing)

- You can run the code with **GPU** on Google Colab, please refer to [the colab tutorial above](#colab-tutorial) for the usage of it.

## **Grading**

Part 1: *Train & Run MobileNet Classifier*
- Setup - 5%
- Data - 5%
- Model - 10%
- Optimization - 10%
- Training - 25%
- Export Model - 5%
- Inference - 10%


Part 2: *LLM with torch.compile*
- Loading LLM - 20%
- Inference with torch.compile - 10%

:::info
**Note: 
Your accuracy of image classification task must be higher than 92.5%. 
Higher accuracy (e.g. 95%) doesn't result in higher score.**
:::

## **Hand-In Policy**
- **YourID.zip**
    - YourID.ipynb
    - YourID_acc_1.png **(eval after model trained in part 1)**
    - YourID_acc_2.png **(eval after model loaded in part 1)**
    - YourID_onnx.png
    - YourID_speedup.png **(part 2, Your Speedup must be greater than 1.5)**

## **Penalty**

- Wrong Format - 10%
- Late Submission - 10% per day