# LLM Fine-Tuning Project: Sentiment Analysis with DistilBERT and LoRA

This repository contains the code and data for a project focused on fine-tuning a Large Language Model (LLM) for sentiment analysis on the Stanford Sentiment Treebank (SST-2) dataset. The project demonstrates dataset preparation, model selection, fine-tuning using Parameter-Efficient Fine-Tuning (PEFT) with LoRA, hyperparameter optimization, comprehensive evaluation, and error analysis.


## 1. Project Overview

The primary goal of this project is to enhance the performance of a pre-trained LLM for binary sentiment classification (positive/negative). We fine-tune a `distilbert-base-uncased` model on the SST-2 dataset using the Hugging Face `Trainer` API and LoRA, a parameter-efficient fine-tuning technique. The project includes a robust evaluation methodology, comparison against a baseline model, and detailed error analysis to understand model limitations.

## 2. Dataset

The dataset used for this project is the Stanford Sentiment Treebank (SST-2), a widely recognized benchmark for sentiment analysis. It consists of movie reviews labeled as positive or negative.

The data files, as structured in Google Colab, will be present in your repository under a `sample_data/` directory.
- `sample_data/`
    - `README.md`
    - `anscombe.json`
    - `california_housing_test.csv`
    - `california_housing_train.csv`
    - `mnist_test.csv`
    - `mnist_train_small.csv`

**Note:** The `LLM_Fine_Tuning_Project.ipynb` notebook automatically downloads the `glue` dataset, which includes SST-2. The `sample_data` directory contains unrelated sample data often found in Colab environments, but it's not directly used by this project's fine-tuning script.

## 3. Setup Instructions

To set up the environment and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HiteshSonetaNEU/LLM-Fine-tuning.git
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The notebook uses an in-notebook `pip install` command, but for a local setup, you can generate a `requirements.txt` file from the notebook's installed packages. For convenience, here's a basic `requirements.txt` based on the notebook:

    
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU (Optional but Recommended):**
    This project heavily benefits from a GPU. Ensure you have CUDA installed and configured if running locally with a GPU. Google Colab instances automatically provide a GPU when selected.

## 4. How to Run the Notebook

The core of this project is the `LLM_Fine_Tuning_Project.ipynb` Jupyter Notebook.

1.  **Open the notebook:**
    You can open it in Google Colab (recommended for GPU access) or locally with JupyterLab/Jupyter Notebook.

    * **Google Colab:** Go to `File > Upload notebook` and select `LLM_Fine_Tuning_Project.ipynb`.
    * **Local Jupyter:** Navigate to the project directory in your terminal and run `jupyter lab` or `jupyter notebook`.

2.  **Run all cells:**
    Execute all cells in the notebook sequentially from top to bottom.
    - **In Colab:** `Runtime > Run all`
    - **In Jupyter:** `Cell > Run All`

    The notebook will:
    - Install necessary libraries.
    - Download and preprocess the SST-2 dataset.
    - Load and configure the DistilBERT model with LoRA.
    - Train the model using three different hyperparameter configurations.
    - Evaluate the best performing fine-tuned model against a baseline.
    - Perform error analysis.
    - Demonstrate an inference pipeline.
    - Generate performance plots and a JSON summary of results.

## 5. Project Approach & Implementation Highlights

* **Dataset:** SST-2 for binary sentiment classification. Data is loaded using the `datasets` library, preprocessed (whitespace removal, short sentence filtering), and split into train, validation, and test sets.
* **Model:** `distilbert-base-uncased` is selected due to its balance of performance and efficiency, making it suitable for fine-tuning with limited computational resources.
* **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA (`r=8`, `lora_alpha=32`, `lora_dropout=0.1`) is applied to reduce the number of trainable parameters, speeding up training and reducing memory footprint.
* **Training Framework:** Hugging Face `Trainer` API is used for its high-level abstraction and efficient training loop management, including logging and checkpointing.
* **Hyperparameter Optimization:** A grid search approach is used to test three distinct combinations of learning rates and epochs, with the best model selected based on validation F1-score.
* **Evaluation:** Comprehensive metrics including Accuracy, Precision, Recall, F1-score (macro and weighted), and AUC-ROC are used. A detailed comparison against the pre-fine-tuned baseline model is performed, including a statistical significance test (McNemar's test).
* **Error Analysis:** Misclassified examples are analyzed, and patterns in errors (e.g., sarcasm, mixed sentiment) are identified to inform future improvements.

## 6. Results & Performance

After running the notebook, the `fine_tuning_results.json` file will contain a detailed summary of the hyperparameter experiments, final performance metrics, and error analysis.

**Key Findings:**

* Significant improvement in sentiment classification performance compared to the baseline `distilbert-base-uncased` model.
* The fine-tuned model achieved [Insert best F1/Accuracy values here from your `fine_tuning_results.json` or notebook output, e.g., `~0.8851 Accuracy` and `~0.8850 F1 Score (Macro)`].
* The comparison showed an improvement of approximately [+XX% in Accuracy, +YY% in F1 Score (Macro)] over the baseline model.
* Visualizations: `hyperparameter_comparison.png` shows the performance of different configurations, and `confusion_matrix.png` provides insight into true vs. predicted labels.

## 7. Error Analysis Insights

The error analysis section in the notebook identifies common patterns in misclassifications, such as:

* **Subtle Negative Words:** Sentences containing nuanced negative indicators or double negatives that the model struggles to interpret.
* **Mixed Sentiment:** Reviews that contain both positive and negative elements, leading to ambiguous classification.
* **Contextual Positives:** Instances where positive words are used in an ironic or negative context.
* **Sarcasm/Irony:** The model faces challenges in discerning sarcastic or ironic statements.

Based on these patterns, the notebook provides actionable suggestions for further model improvement, including data augmentation strategies, architecture enhancements, and training techniques.

## 8. Inference Pipeline

The `LLM_Fine_Tuning_Project.ipynb` notebook includes a `SentimentAnalyzer` class that provides a simple interface to load and use the fine-tuned model for prediction on new text inputs. This demonstrates how the fine-tuned model can be deployed for practical use.

