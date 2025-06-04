
# 📝 Formalizacja Tekstu w Języku Polskim

## 📌 Opis projektu

Celem projektu jest stworzenie systemu, który automatycznie przekształca nieformalne wypowiedzi w języku polskim na ich formalne odpowiedniki. Tego typu narzędzie znajduje zastosowanie m.in. w środowiskach akademickich, zawodowych i administracyjnych, gdzie konieczne jest zachowanie profesjonalnego tonu wypowiedzi.

Rozwiązanie opiera się na uczeniu maszynowym – przygotowaliśmy syntetyczny zbiór danych (pary: zdanie nieformalne – zdanie formalne), który posłużył do trenowania i ewaluacji modelu językowego.

Syntetyczny korpus zdań został wygenerowany z użyciem dostępnych dużych modeli językowych (LLM), w tym m.in. modeli instrukcyjnych w trybie chatowym. Umożliwiło to szybkie pozyskanie dużej liczby przykładów o wysokiej jakości językowej, co znacząco wpłynęło na skuteczność procesu fine-tuningu.

Modele były trenowane i ewaluowane w środowisku Google Colab, co zapewniło elastyczność oraz łatwy dostęp do zasobów GPU, pozwalając na efektywne przeprowadzenie eksperymentów bez konieczności konfiguracji lokalnego środowiska obliczeniowego.

Model został udostępniony poprzez REST API oraz zintegrowany z aplikacją webową stworzoną w Streamlit. Interfejs użytkownika zawiera również komponent feedbackowy umożliwiający ocenę jakości predykcji (thumbs up/down), który zapisuje dane do bazy.

Szczegółowy opis eksperymentów (fine-tuning, metryki, porównania modeli) jest dostępny w systemie MLflow:  
🔗 [Zobacz eksperymenty w MLflow](https://dagshub.com/informal2formal/mlflow/experiments)

---

## ⚙️ Funkcje aplikacji

- 🔄 Automatyczna formalizacja tekstu (z nieformalnego na formalny)
- 🤖 Hostowanie wytrenowanego modelu na Hugging Face Hub
- 🌐 REST API zintegrowane z frontendem (Streamlit)
- 👍👎 Komponent feedbacku (zapisywanie opinii użytkownika do bazy danych)
- 📈 Metryki ewaluacji BLEU / ROUGE dostępne w MLflow
- 🔒 Obsługa błędów i walidacja danych wejściowych

---

## 🚀 Instrukcja uruchomienia

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/yanvoi/informal_to_formal_llm.git
cd informal_to_formal_llm
```

### 2. Uruchomienie środowiska

Instrukcje dotyczące uruchomienia środowiska znajdują się w pliku `README.md` w folderze `app`.

---

## 📁 Struktura projektu

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

## 📄 Autorzy

- Jan Wojciechowski – 473553  
- Sebastian Jerzykiewicz – 473615  
- Jędrzej Rybczyński – 456532
