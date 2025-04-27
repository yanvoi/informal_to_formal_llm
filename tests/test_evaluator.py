import pandas as pd
import pytest

from informal_to_formal.evaluation import Evaluator


@pytest.fixture()
def mock_validation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pred": [
                "Popełniłem błąd w raporcie.",
                "Nie wiem, co się stało.",
                "To jest test jednostkowy.",
            ],
            "target": [
                "Popełniłem błąd w raporcie.",
                "Nwm, co się wydarzyło.",
                "To są testy integracyjne.",
            ],
        }
    )


def test_evaluator_validation(mock_validation_data):
    mock_validation_data.rename(columns={"pred": "invalid_col_name"}, inplace=True)

    with pytest.raises(ValueError, match="Missing required column: 'pred'"):
        Evaluator(mock_validation_data)


def test_evaluate(mock_validation_data):
    evaluator = Evaluator(mock_validation_data)
    df_all, avg_all = evaluator.evaluate()

    assert df_all.shape[0] == mock_validation_data.shape[0]

    # Check if all evaluation columns are present
    evaluation_columns = ["rouge1", "rouge2", "rougeL", "bert_precision", "bert_recall", "bert_f1"]
    assert all(col in df_all.columns for col in evaluation_columns)

    # Check if the average scores are returned
    assert isinstance(avg_all, dict)
    assert all(col in avg_all for col in evaluation_columns)
    assert all(isinstance(avg_all[col], float) for col in evaluation_columns)
