# Fine Tuning LLM (Using Tensorflow)

## Enhancing Financial Sentiment Analysis with Fine-Tuned DistilBERT

### Overview
This project showcases the process of fine-tuning the DistilBERT model to perform sentiment analysis on financial text data. The aim is to enhance the model's ability to accurately classify sentiments within financial documents by experimenting with different training techniques, learning rates, and strategies to handle class imbalances. Through detailed error analysis and multiple experimental setups, this project achieves high accuracy and provides valuable insights into optimizing large language models for specific tasks in the financial domain.

### Run on Google Colab
You can also run the project on Google Colab using the following link: [Colab Notebook](https://colab.research.google.com/drive/18egCEmEa5H4ablxko7eVpdYMe-as15NQ?usp=sharing).

**Note:** Enable GPU for training the model. The free GPU provided by Colab will work.

### Dataset Structure

Each record in the dataset consists of:

- **Text**: Financial news or corporate announcements.
- **Sentiment**: The sentiment label — `0` for Negative, `1` for Neutral, and `2` for Positive.

#### Dataset Fields:

- **ID**: Unique identifier for each entry.
- **Text**: The financial news or report.
- **Sentiment**: Sentiment label (`0`, `1`, `2`).

#### Sentiment Labels:

- `0`: Negative
- `1`: Neutral
- `2`: Positive

#### Example Entries

| ID  | Text                                                                                       | Sentiment |
|-----|---------------------------------------------------------------------------------------------|-----------|
| 1   | "The current lay-offs are additional to the temporary lay-offs agreed in December 2008 and in May 2009." | 0 (Negative) |
| 2   | "The acquisition is expected to take place by the end of August 2007."                      | 1 (Neutral)  |
| 3   | "Strong brand visibility nationally and regionally is of primary importance in home sales, vehicle and consumer advertising." | 1 (Neutral)  |


### Key Features
- **Data Loading and Preprocessing**: Utilizing the `datasets` library to load the financial phrasebank dataset and preprocessing it for model training.
- **Model Training**: Experimenting with different approaches like training only the classifier head, fine-tuning the entire model, and employing various learning rate strategies.
- **Class Imbalance Handling**: Implementing techniques to address class imbalance in the dataset.
- **Performance Evaluation**: Evaluating the model using classification reports, confusion matrices, and conducting error analysis.
- **Experiments with Dataset Variants**: Testing the model on different versions of the financial phrasebank dataset to compare performance.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/adibakshi28/DL-Fine-Tune_LLM-Tensorflow.git
    cd DL-Fine-Tune_LLM-Tensorflow
    ```

2. Install the required packages:
    ```bash
    pip install tensorflow transformers datasets jupyter nbformat
    ```

### Usage
1. Open the Jupyter Notebook:
    ```bash
    jupyter notebook Deep_Learning_Project.ipynb
    ```

2. Follow the steps in the notebook to:
    - Load and preprocess the financial data.
    - Fine-tune the DistilBERT model.
    - Evaluate and analyze the model's performance.

### Detailed Project Content
- **Basic Assignment 1**: Training only the classifier head using DistilBERT with default Adam optimizer settings, evaluating the performance on training, validation, and test sets.
- **Basic Assignment 2**: Fine-tuning all layers of the DistilBERT model except the pre-trained classifier head, leveraging the previously trained head for better performance.
- **Basic Assignment 3**: Fine-tuning the entire model, including both the pre-trained classifier head and the DistilBERT base layers, to enhance the model's understanding of financial sentiment.
- **Experiment 1**: Creating a TensorFlow dataset from the tokenized data, illustrating the conversion and preparation for training.
- **Experiment 2**: Conducting superior error analysis with detailed classification reports and confusion matrices, addressing issues like class imbalance and model bias.
- **Experiment 3**: Implementing class weights to handle imbalanced data, improving model performance on underrepresented classes.
- **Experiment 4**: Evaluating the model on different variants of the financial phrasebank dataset, comparing results across various agreement levels of the dataset.
- **Experiment 5**: Developing a custom classification head for the DistilBERT model, showcasing the flexibility in adapting the model architecture for specific tasks.

### Results
The project achieves significant improvements in sentiment classification accuracy on financial texts, with experiments demonstrating the effectiveness of various training and optimization techniques.

### Contributing
Feel free to submit issues or pull requests for improvements or new features.

### Contact
For questions or suggestions, please feel free to contact me or raise an issue on the repo.
