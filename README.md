
# 📝 Text Formalization in the Polish Language

## 📌 Project Description

The goal of this project is to create a system that automatically transforms informal Polish utterances into their formal equivalents. Such a tool is particularly useful in academic, professional, and administrative contexts, where maintaining a professional tone is essential.

The solution is based on machine learning – we prepared a synthetic dataset (pairs: informal sentence – formal sentence) used to train and evaluate the language model.

The synthetic corpus was generated using available large language models (LLMs), including instruction-tuned models in chat mode. This enabled the rapid creation of a large number of high-quality linguistic examples, which significantly improved the effectiveness of the fine-tuning process.

The models were trained and evaluated in the Google Colab environment, providing flexibility and easy access to GPU resources. This allowed efficient experimentation without the need to configure a local computing environment.

The trained model is available via a REST API and integrated with a web application built using Streamlit. The user interface also includes a feedback component (thumbs up/down), which stores user ratings in a database.

A detailed description of the experiments (fine-tuning, metrics, model comparisons) is available in the MLflow system:
🔗 [Zobacz eksperymenty w MLflow](https://dagshub.com/informal2formal/mlflow/experiments)

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
│   ├── api/                 # Katalog z kodem API
│   │   ├── __init__.py      # Inicjalizacja pakietu
│   │   ├── main.py          # Główny plik API
│   ├── ui/                  # Katalog z kodem interfejsu użytkownika
│   │   ├── main.py          # Główny plik aplikacji Streamlit
│   ├── README.md            # Instrukcje uruchomienia aplikacji
├── informal_to_formal/
│   ├── data_preprocessor/   # Katalog z kodem do przetwarzania danych
│   ├── evaluation/          # Katalog z kodem do ewaluacji modelu
│   ├── training/            # Katalog z kodem do trenowania modelu
│   ├── utils/               # Katalog z pomocniczymi funkcjami
│   ├── __init__.py          # Inicjalizacja pakietu
├── notebooks/               # Katalog z notatnikami Jupyter do trenowania i ewaluacji modeli
├── tests/                   # Katalog z testami jednostkowymi
├── README.md                # Główny plik README projektu (ten plik)
```

---

## 📄 Authors

- Jan Wojciechowski – 473553  
- Sebastian Jerzykiewicz – 473615  
- Jędrzej Rybczyński – 456532
