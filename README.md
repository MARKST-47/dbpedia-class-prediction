# DBPedia Text Classification with Custom Neural Network

This project focuses on exploratory data analysis (EDA) and text classification using the DBPedia dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/danofer/dbpedia-classes). The task involves classifying text into hierarchical categories (l1, l2, l3), which represent different levels of classification. We utilize a custom neural network, including LSTM layers, to predict these hierarchical labels.

## Table of Contents

1. [Dataset](#dataset)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Modeling](#modeling)
4. [Training and Evaluation](#training-and-evaluation)
5. [Prediction](#prediction)
6. [Requirements](#requirements)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [License](#license)

## Dataset

The dataset used in this project is the DBPedia Classes dataset, which contains text data along with hierarchical labels representing different levels of classification:

- **l1:** Top-level category (e.g., Agent)
- **l2:** Sub-category of l1 (e.g., Politician)
- **l3:** Sub-category of l2 (e.g., Senator)

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/danofer/dbpedia-classes).

The dataset includes:
- `DBPEDIA_train.csv`: Training data
- `DBPEDIA_test.csv`: Test data
- `DBPEDIA_val.csv`: Validation data

## Exploratory Data Analysis (EDA)

Performed EDA to better understand the dataset, including the following steps:

1. **Text Preprocessing:**
   - Lowercasing
   - Removing numbers
   - Removing punctuation
   - Removing extra spaces

2. **Word Cloud Visualization:**
   - Visualized the most frequent words in the dataset after removing stop words.

3. **Label Distribution:**
   - Visualized the distribution of l1, l2, and l3 labels.

## Modeling

Built a custom neural network model (BERT model will work better but takes much longer to train - also included in prediction notebook) with the following architecture:

- **Embedding Layer:** Converts input tokens into dense vectors of fixed size.
- **LSTM Layer:** Processes the sequence data, capturing long-term dependencies.
- **Fully Connected Layer:** Maps the LSTM output to the label space.
- **Softmax Activation:** Outputs probabilities for each class.

### Loss Function and Optimizer
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam

## Training and Evaluation

The model was trained for 3 epochs, and progress was tracked using the `tqdm` library to show real-time progress bars.


## Prediction

To predict the labels for a new sentence, the model processes the text and outputs the most probable hierarchical labels (l1, l2, l3).

### Example Usage:
```python
sentence = "This is an example sentence to classify."
prediction = predict_sentence(sentence, model, vocab, maxlen)
print(f"This sentence belongs to: {prediction}")
```

## Requirements

- Python 3.7+
- torch
- numpy
- pandas
- scikit-learn
- tqdm
- matplotlib
- seaborn

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dbpedia-text-classification.git
   cd dbpedia-text-classification
   ```

2. Download the DBPedia dataset from [Kaggle](https://www.kaggle.com/datasets/danofer/dbpedia-classes) and place it in a folder named `DBPEDIA/`.

3. Run the EDA notebook to explore the dataset:
   ```bash
   jupyter notebook DBpedia_EDA.ipynb
   ```

4. Run the training script to train the model:
   ```bash
   jupyter notebook DBpedia_predictions.ipynb
   ```

5. Use the trained model for predictions as shown in the prediction section.

## Results

The model achieved reasonable accuracy in predicting the hierarchical labels (l1, l2, l3) on the validation set. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
