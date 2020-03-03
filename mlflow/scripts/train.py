import argparse
import os
from urllib.parse import urlparse

import allennlp
from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.common.util import import_submodules
import mlflow
import mlflow.pyfunc
from mlflow.utils.file_utils import yaml


class AllennlpPredictorWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, predictor_name: str = None):
        self._predictor_name = predictor_name

    def load_context(self, context):
        from allennlp.predictors import Predictor
        self.predictor = Predictor.from_path(
            context.artifacts["model_archive"],
            predictor_name=self._predictor_name,
        )

    def predict(self, context, model_input):
        inputs = model_input.to_dict(orient="records")
        return self.predictor.predict_batch_json(inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("param_path", type=str)
    parser.add_argument("-o", "--overrides", type=str, default="")
    parser.add_argument("--include-package", type=str, action="append", default=[])
    parser.add_argument("--predictor", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="allennlp_model")
    parser.add_argument("--conda-env", type=str, default="conda.yaml")
    args = parser.parse_args()

    for package_name in args.include_package:
        import_submodules(package_name)

    params = Params.from_file(args.param_path, args.overrides)

    with mlflow.start_run():
        artifact_uri = urlparse(mlflow.get_artifact_uri())
        if artifact_uri.scheme != "file":
            raise RuntimeError("scheme not supported: {artifact_uri.scheme}")

        serialization_dir = artifact_uri.path
        _model = train_model(params, serialization_dir)

        artifacts = {
            "model_archive": mlflow.get_artifact_uri("model.tar.gz")
        }
        with open(args.conda_env, "r") as f:
            conda_env = yaml.safe_load(f)

        mlflow.pyfunc.log_model(
            artifact_path=args.model_path,
            python_model=AllennlpPredictorWrapper(args.predictor),
            artifacts=artifacts,
            conda_env=conda_env,
        )
