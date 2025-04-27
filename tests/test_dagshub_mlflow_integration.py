import os
import mlflow
import pytest
import dagshub

from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def initialize_dagshub():
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")

    assert dagshub_token, "DAGSHUB_TOKEN not found in .env"
    assert repo_owner, "DAGSHUB_REPO_OWNER not found in .env"
    assert repo_name, "DAGSHUB_REPO_NAME not found in .env"

    # Initialize Dagshub connection
    dagshub.auth.add_app_token(token=dagshub_token)
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


@pytest.fixture()
def get_example_experiment_with_run():
    return "test_experiment", "test_iteration"


def test_experiment_and_run_exist(get_example_experiment_with_run):
    experiment_name, run_name = get_example_experiment_with_run

    # Initialize the client
    client = mlflow.tracking.MlflowClient()

    # Fetch the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None

    # Check if the specific run name exists
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    run_names = [run.data.tags.get("mlflow.runName") for run in runs]
    assert run_name in run_names

    # Check if the run is properly logged
    run = [run for run in runs if run.data.tags.get("mlflow.runName") == run_name][0]
    assert run.data.metrics["accuracy"] == 42
    assert run.data.params["Param name"] == "Value"
