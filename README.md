# Histopathologic Cancer Detection Deep Learning Model

## Project Overview
This project applies deep learning techniques to detect cancer in histopathologic images of breast cancer tissue. Using convolutional neural networks (CNNs), the model analyzes microscopic images to identify the presence of cancer cells, achieving high accuracy in classification.

## Author
- **Michael Hansen**
- **Course**: DTSA5511 Deep Learning
- **Instructor**: Dr. Ying Sun

## Dataset
The dataset comes from the [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data) Kaggle competition, originally obtained from the [TCGA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) project. It consists of:
- Approximately 220,000 labeled images for training
- About 57,000 images for testing
- Each image is labeled as either containing cancer tissue (1) or not (0)
- A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue

## Project Structure
The project follows these key steps:
1. **Data Exploration and Preparation**
   - Loading and examining the dataset
   - Visualizing sample images and their distributions
   - Analyzing RGB channel distributions between positive and negative samples
   
2. **Model Development**
   - **Initial Model**: A simple CNN with a single convolutional layer, pooling layer, and dense layers
   - **Advanced Model**: A deeper architecture inspired by VGG-16, with multiple convolutional blocks, batch normalization, and regularization techniques
   
3. **Training and Optimization**
   - Data augmentation to improve generalization
   - Implementation of learning rate scheduling
   - Use of callbacks for early stopping and model checkpointing
   - Hyperparameter tuning to improve performance

4. **Evaluation and Results**
   - Achieved validation accuracy of approximately 80-85%
   - Consistent performance between training and validation sets
   - Final Kaggle submission score of 0.893

## Technical Implementation
The project utilizes:
- **TensorFlow/Keras** for model building and training
- **OpenCV** for image processing
- **Pandas/NumPy** for data manipulation
- **Matplotlib** for visualization
- **Scikit-learn** for data splitting and evaluation metrics

### Model Architecture
The final model architecture includes:
- Multiple convolutional blocks with increasing filter sizes (32 → 64 → 128)
- Batch normalization after each convolutional layer
- Max pooling layers to reduce dimensionality
- Dropout layers to prevent overfitting
- L2 regularization to improve generalization
- Dense layers with appropriate activation functions

## Key Findings
- Deep learning models can effectively identify cancer in histopathologic images
- Data augmentation significantly improves model performance
- Batch normalization and regularization techniques help prevent overfitting
- Learning rate scheduling improves training stability and final performance
- The model demonstrates strong potential for assisting pathologists in cancer detection

## Conclusion
This project demonstrates the power of deep learning in medical image analysis. The developed CNN model successfully identifies cancer in histopathologic images with high accuracy, showing potential for real-world applications in assisting medical professionals with cancer diagnosis.

The implementation of advanced techniques like data augmentation, batch normalization, and learning rate scheduling proved crucial in achieving robust performance. Future work could explore ensemble methods, more complex architectures, or transfer learning approaches to further improve accuracy.

## References
- VGG-16 CNN Model: https://www.geeksforgeeks.org/vgg-16-cnn-model/
- Kaggle Competition: https://www.kaggle.com/competitions/histopathologic-cancer-detection/data
- TCGA Project: https://portal.gdc.cancer.gov/projects/TCGA-BRCA 
