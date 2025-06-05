# 📝 Text Formalization in the Polish Language

## 📌 Project Description

The goal of this project is to create a system that automatically transforms informal Polish utterances into their formal equivalents. Such a tool is particularly useful in academic, professional, and administrative contexts, where maintaining a professional tone is essential.

The solution is based on machine learning – we prepared a synthetic dataset (pairs: informal sentence – formal sentence) used to train and evaluate the language model.

The synthetic corpus was generated using available large language models (LLMs), including instruction-tuned models in chat mode. This enabled the rapid creation of a large number of high-quality linguistic examples, which significantly improved the effectiveness of the fine-tuning process.

The models were trained and evaluated in the Google Colab environment, providing flexibility and easy access to GPU resources. This allowed efficient experimentation without the need to configure a local computing environment.

The trained model is available via a REST API and integrated with a web application built using Streamlit. The user interface also includes a feedback component (thumbs up/down), which stores user ratings in a database.

A detailed description of the experiments (fine-tuning, metrics, model comparisons) is available in the MLflow system:
🔗 [Check the experiments on MLflow](https://dagshub.com/informal2formal/mlflow/experiments)

## 📄 Dataset
The synthetic dataset used for training and evaluation is publicly available on DagsHub:

🔗 [Check the generated dataset on DagsHub](https://dagshub.com/informal2formal/mlflow/datasets)

It contains pairs of informal and formal Polish sentences generated using instruction-tuned large language models.

---

## 🤖 Hugging Face Model

The final trained model is publicly available on the 🤗 **Hugging Face Hub**:

🔗 [directtt/Llama-PLLuM-8B-instruct-informal2formal-SFT](https://huggingface.co/directtt/Llama-PLLuM-8B-instruct-informal2formal-SFT)

In the repository, you can find the model card with detailed information about the training process, evaluation metrics, and usage instructions.

---

## ⚙️ Application Features

- 🔄 Automatic text formalization (from informal to formal)
- 🤖 Hosting of the trained model on Hugging Face Hub
- 🌐 REST API integrated with the frontend (Streamlit)
- 👍👎 Feedback component (saves user ratings to a database)
- 📈 Evaluation metrics BLEU / ROUGE available in MLflow
- 🔒 Error handling and input data validation

---

## 🚀 Launch Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yanvoi/informal_to_formal_llm.git
cd informal_to_formal_llm
```

### 2. Launch the environment

Instructions for setting up the environment can be found in the `README.md` file in the `app` folder.


🛠️ Note: All Python module versions and dependencies are listed in the pyproject.toml file.

---

## 📁 Project Structure

```
informal_to_formal_llm/
│
├── app/
│   ├── api/                 # API code
│   │   ├── __init__.py
│   │   ├── main.py
│   ├── ui/                  # Streamlit frontend
│   │   ├── main.py
│   ├── README.md            # App-specific setup instructions
├── informal_to_formal/
│   ├── data_preprocessor/   # Data preprocessing scripts
│   ├── evaluation/          # Model evaluation tools
│   ├── training/            # Model training scripts
│   ├── utils/               # Helper functions
│   ├── __init__.py
├── notebooks/               # Jupyter notebooks for experimentation
├── tests/                   # Unit tests
├── pyproject.toml           # Project dependencies and configuration
├── README.md                # Main project README (this file)
```

---

## 🧠 Techniques Used

### Few-Shot Prompting
Few-shot prompting was employed to guide the large language models (LLMs) in generating high-quality synthetic data. By providing a few examples of informal-to-formal transformations, the models were able to generalize and produce a diverse set of sentence pairs.

### Fine-Tuning
The LLMs were fine-tuned on the synthetic dataset to specialize in the task of text formalization. This process involved adjusting the pre-trained weights of the models to better align with the specific task requirements.

### Quantization
Quantization techniques were applied to reduce the size of the trained models, making them more efficient for deployment without significant loss in performance. This step was crucial for integrating the models into the web application and REST API.

### Evaluation Metrics
The performance of the models was evaluated using BLEU and ROUGE metrics. These metrics provided insights into the quality of the formalized text compared to the reference sentences.

### Data Augmentation
Synthetic data generation was a key component of the training process. By leveraging instruction-tuned LLMs, a large and diverse dataset was created, which significantly improved the robustness of the models.

---

## 📄 Authors

- Jan Wojciechowski – 473553  
- Sebastian Jerzykiewicz – 473615  
- Jędrzej Rybczyński – 456532

## 🔚 Conclusions

- We did not use the GYAFC corpus, despite our efforts to obtain it, for the following reasons:
  1. Translating it to Polish would have been a significant challenge.
  2. The quality of the corpus was rather poor, with sentences often being ambiguous.

- Unsloth provided a better interface for fine-tuning and utilizing large language models (LLMs), making it our preferred choice.

- The performance of the trained models improved with an increasing number of epochs, but the improvements plateaued after two epochs, showing no significant gains beyond that point.
