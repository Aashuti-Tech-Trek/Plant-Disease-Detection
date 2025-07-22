# **Plant Disease Classification using CNN and Transfer Learning**

---

## **Objective**

Agricultural productivity is highly dependent on timely and accurate diagnosis of plant diseases. Traditional manual inspection by experts is often labor-intensive, inconsistent, and impractical at scale. Especially in rural areas with limited access to agricultural specialists.

This project aims to develop an **automated plant disease classification system** using deep learning techniques on leaf images. The primary goals are to:

- Detect a wide variety of plant diseases with high accuracy.  
- Enable scalable, real-time, and reliable diagnosis using leaf images.  
- Build a lightweight and transferable system suitable for mobile applications.

By leveraging **Convolutional Neural Networks (CNNs)** and **Transfer Learning** models like **ResNet**, **DenseNet**, and a **Hybrid Model**, we enhance performance, generalization, and training efficiency. **Data augmentation** is used to increase robustness and avoid overfitting.

---

## **Methodology**

This project applies a combination of **deep learning** and **transfer learning** techniques as follows:

### **CNN with Transfer Learning**

We used three pretrained models: **ResNet**, **DenseNet**, and a **Hybrid of both**, to classify plant disease images:

-  **ResNet**: A deep residual network that combats vanishing gradients through skip connections, enabling deeper architecture.  
-  **DenseNet**: A densely connected network promoting feature reuse and mitigating gradient issues.  
-  **Hybrid Model**: A custom model combining ResNet and DenseNet to leverage strengths from both architectures.

**Data augmentation** was applied to diversify the dataset and help models generalize better.

---

## **Why These Models?**

### **ResNet**
- Great for deep image classification.
- Residual connections improve gradient flow and training speed.

### **DenseNet**
- Efficient use of parameters by reusing features.
- Each layer connects to every other, encouraging compact yet powerful representations.

### **Hybrid Model**
- Merges feature strengths of ResNet and DenseNet.
- Aims to capture richer and more diverse spatial features.

 **Transfer learning** allows use of pretrained weights from **ImageNet**, reducing training time and improving performance with limited data.

---

## **Results**

> **Reference Paper**: [Integrated Leaf Disease Recognition Across Diverse Crops through Transfer Learning](https://doi.org/10.1016/j.procs.2024.03.192)

| **Model**       | **Source**   | **Accuracy** | **Train Accuracy** | **Val Accuracy** | **Train Loss** | **Val Loss** |
|-----------------|--------------|--------------|---------------------|------------------|----------------|--------------|
| MobileNetV2     | Base Paper   | 91.98%       | –                   | –                | –              | –            |
| DenseNet        | Proposed     | 95.57%       | 92.2%               | 95.2%            | 0.233          | 0.153        |
| ResNet          | Proposed     | 94.49%       | 99.7%               | 99.4%            | 0.721          | 0.715        |
| Hybrid          | Proposed     | 99.36%       | 99.0%               | 92.9%            | 0.747          | 0.905        |

**Observations:**
- **DenseNet** achieved the best generalization with the lowest validation loss.
- **ResNet** displayed the highest training accuracy.
- The **Hybrid model** captured richer features but had slightly higher validation loss.

---

## **Tools and Libraries**

- Python  
- TensorFlow  
- Scikit-learn  
- Matplotlib & Seaborn
- PIL 

---

## **Dataset Used**

- **Name**: *New Plant Diseases Dataset*  
- **Description**:  
  - Contains **87,000+ RGB images** of healthy and diseased leaves across **38 classes**.  
  - Data split: **80% for training**, **20% for validation**.  
  - Includes **33 external test images** for additional evaluation.
- **Link**: [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
---

## **Conclusion**

This project shows that **CNN-based transfer learning** is highly effective for plant disease classification. Among the three architectures, **DenseNet** provided the most balanced performance with high validation accuracy and low loss.

Compared to the base paper, the **proposed models**, especially **DenseNet** and the **Hybrid model**, **outperformed prior benchmarks** and present a robust, deployable solution for real-world agricultural applications.

---

## **Contact**
For queries, collaborations, clarifications, or implementation guidance:
**Aashuti Gambhir**  
  - Email: [atechtrek@gmail.com](mailto:atechtrek@gmail.com)  
  - GitHub: [Aashuti-Tech-Trek](https://github.com/Aashuti-Tech-Trek)
