
---

# Food Recognition and Calorie Estimation

## Project Overview
This project develops a deep learning model to recognize food items from images and estimate their calorie content. It uses the **Fruits 360 Dataset** to train a Convolutional Neural Network (CNN) for food classification. Estimated calorie content for each food item helps users track their dietary intake effectively.

---

## Dataset
The dataset used is the **Fruits 360 Dataset**.

- **Source**: [Fruits 360 Dataset on Kaggle](https://www.kaggle.com/moltean/fruits)
- **Structure**: 
  - Training and testing data are organized into subfolders, each representing a specific fruit class.
  - Images are 100x100 pixels in RGB format.

---

## Workflow

### 1. **Download and Prepare the Dataset**
The dataset is downloaded using `kagglehub` and extracted to the specified directory. The training and testing images are loaded using TensorFlow's `ImageDataGenerator`.

### 2. **Data Preprocessing**
- **Normalization**: Rescaling image pixel values to the range `[0, 1]`.
- **Target Size**: Resizing images to `100x100` pixels to match the model's input requirements.

### 3. **Model Architecture**
The CNN model consists of:
- Convolutional layers for feature extraction.
- Max-pooling layers for dimensionality reduction.
- Dense layers for classification.
- A dropout layer to prevent overfitting.
- A softmax activation function in the output layer for multi-class classification.

### 4. **Model Training**
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Epochs**: 10

### 5. **Evaluation**
The model is evaluated on the test set to compute accuracy and generate predictions for unseen data.

### 6. **Calorie Estimation**
A mapping of food classes to calorie content is defined manually. Based on the model's predictions, calorie values are retrieved and displayed.

### 7. **Visualization**
- Sample test images are displayed alongside their predicted classes and calorie estimates.

---

## Code Details

### Calorie Estimation Logic
The `calorie_map` dictionary maps each food class to its average calorie content per 100g. Predictions are mapped to classes, and their corresponding calorie values are retrieved.

```python
# Example calorie mapping
calorie_map = {
    'Apple': 52,
    'Banana': 89,
    'Blueberry': 57,
    'Carrot': 41,
    'Orange': 47,
    # Add more mappings as needed
}

# Function to estimate calories
def estimate_calories(predictions, class_indices):
    predicted_classes = [list(class_indices.keys())[np.argmax(pred)] for pred in predictions]
    estimated_calories = [calorie_map.get(cls, "Unknown") for cls in predicted_classes]
    return predicted_classes, estimated_calories
```

---

## Results

### Metrics
- **Training Accuracy**: Achieved during training.
- **Test Accuracy**: Displayed after model evaluation.

### Sample Output
Predicted food items and calorie estimates for test images are displayed in a grid.

```text
Food: Apple
Calories: 52

Food: Banana
Calories: 89

...
```

---

## Dependencies

Install required libraries:
```bash
pip install numpy pandas matplotlib tensorflow kagglehub
```

---

## Usage

1. Ensure the dataset is available for download via Kaggle.
2. Run the script:
   ```bash
   python food_recognition_calories.py
   ```
3. The script will train the CNN model, evaluate its performance, and display calorie estimates for test samples.

---

## Future Enhancements

1. **Add More Food Items**: Expand the dataset to include a wider variety of foods.
2. **Improve Calorie Mapping**: Incorporate a more comprehensive and dynamic calorie database.
3. **Real-Time Predictions**: Enable real-time food recognition from live camera feeds using OpenCV.
4. **Nutritional Analysis**: Extend the system to include other nutritional information like protein, carbs, and fats.

