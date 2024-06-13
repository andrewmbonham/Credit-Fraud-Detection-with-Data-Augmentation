import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

import pickle


class Generator(nn.Sequential):
    def __init__(self, sample_size: int):
        """Generator for synthetic fraudulent credit transactions

        Args:
            sample_size: integer number of samples to generate
        """
        super().__init__(
            nn.Linear(sample_size, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 29),
        )

        # Random value vector size
        self.sample_size = sample_size

    def forward(self, batch_size: int):
        # Generate randon values
        z = torch.randn(batch_size, self.sample_size)

        # Generator output
        output = super().forward(z)

        # Convert the output shape from (batch_size, 29) to (batch_size, 1, 29)
        generated_transactions = output.reshape(batch_size, 1, 29)
        return generated_transactions


class Discriminator(nn.Sequential):
    def __init__(self):
        """Discriminator to detect synthetic transaction data."""
        super().__init__(
            nn.Linear(29, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1)
        )  # change dimensions

    def forward(self, transactions: torch.Tensor, targets: torch.Tensor):
        prediction = super().forward(transactions.reshape(-1, 29))  # change dimensions
        loss = F.binary_cross_entropy_with_logits(prediction, targets)
        return loss


def generate_transactions(generator, batch_size=100):
    """Function that uses the given generator to contruct a pandas
        DataFrame of the synthetic transaction data.

    Args:
        generator: class Generator object
        batch_size: integer number of samples to generate

    Returns:
        generated_trasactions: pandas DataFrame of synthetic
            fraudulent transaction data
    """
    with torch.no_grad():
        generated_transactions = generator(batch_size).numpy()
    generated_transactions = pd.DataFrame(
        generated_transactions.reshape(batch_size, 29)
    )
    return generated_transactions


def load_and_describe(file_name):
    """Function to load and summarize credit data from a .csv file.

    Args:
        file_name: string file_name of .csv containing credit transaction data.

    Returns:
        data: pandas DataFrame of unprocessed credit transaction data.
    """
    data = pd.read_csv(file_name)
    hash_str = "#" * 80
    print(hash_str)
    print(data.info())
    print(hash_str)
    print(data.describe())
    print(hash_str)
    return data


def preprocess_data(data):
    """Preprocesses credit data.

    Args:
        data: pandas DataFrame of unprocessed credit transaction data.

    Returns:
        data: normalized DataFrame with time and null values removed.
    """
    #  check for null values
    if data.isna().sum().sum() == 0:
        print("No NaNs to clean.")
    else:
        data.dropna(inplace=True)
    # drop time axis
    try:
        print("Dropping time axis...")
        data.drop("Time", axis=1, inplace=True)
        print("Time axis dropped.")
    except:
        print("Failed to drop time axis. Check column names.")
    # normalize
    print("Normalizing...")
    means, stds = data.iloc[:, :-1].mean(), data.iloc[:, :-1].std()
    data.iloc[:, :-1] = (data.iloc[:, :-1] - means) / stds
    print("Data normalized.")
    return data


def split_by_class(data):
    """Split pandas DataFrame of transaction data by class

    Args:
        data: pandas DataFrame of transaction data

    Returns:
        fraudulent: pandas DataFrame consisting of data["Class"] == 1 rows
        legitimate: pandas DataFrame consisting of data["Class"] == 0 rows
    """
    fraudulent, legitimate = data[data["Class"] == 1], data[data["Class"] == 0]
    frac_fraud = data["Class"].mean()
    print(f"No. fraudulent transactions: {len(fraudulent)}")
    print(f"No. legitimate transactions: {len(legitimate)}")
    print(f"Percentage of fraudulent transactions: {100 * frac_fraud:.3f}%")
    return fraudulent, legitimate


def build_models():
    """Builds untrained models:
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        GradientBoostingClassifier,
            all from sklearn.

    Returns:
        models: dictionary object, where the keys are the model
            names, and the values are the untrained models.
    """
    seed = 0
    models = {
        "Logistic Regression": LogisticRegression(random_state=seed),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=seed
        ),
        "Support Vecctor Machine": SVC(max_iter=100000, random_state=seed),
        "Gradient Boosting Machine": GradientBoostingClassifier(
            max_leaf_nodes=128, random_state=seed
        ),
    }
    return models


def get_models(extension=None):
    """Either loads saved models or builds untrained models
        using build_models()

    Args:
        extension: file extension (depends on augment type typically),
            default is None.

    Returns:
        models: dictionary containing pre-saved models if models file
            with given extenstion is found, untrained models otherwise.
    """

    pretrained = False
    if extension is not None:
        try:
            file_name = f"models.{extension}"
            print("Attempting to load pre-trained models...")
            models = load_models(file_name)
            pretrained = True
            model_list = "\n".join(list(models.keys()))
            outstr = f"The following pretrained models were successfully loaded:\n{model_list}\n"
            print(outstr)
            return models, pretrained
        except:
            print("Loading pre-trained models failed.\nBuilding new models...")
            pass
    models = build_models()
    model_list = "\n".join(list(models.keys()))
    outstr = f"The following models are ready for training:\n{model_list}\n"
    print(outstr)
    return models, pretrained


def load_models(file_name):
    """Loads models saved via pickle.

    Args:
        file_name: string file name of saved models

    Returns:
        result of pickle.load appled to the file
    """
    with open(file_name, "rb") as handle:
        return pickle.load(handle)


def save_models(models, extension):
    """Saves models via pickle.

    Args:
        models: dict object containing model names and the [trained] models
        extension: file extension (file saved is f"models.{extension}")

    Returns:
        result of pickle.dump appled to the file
    """
    if extension is not None:
        file_name = f"models.{extension}"
    else:
        file_name = "credit_fraud_models"
    with open(file_name, "wb") as handle:
        return pickle.dump(models, handle)


def data_check(pretrained=False, X_train=None, y_train=None, X_test=None, y_test=None):
    """Runs preliminary check on data passed to build_and_evaluate(...).

    The build_and_evaluate function can accept None values for some
        data objects, but not all. This function ensures that if the
        models are not pretrained, then there is training data, and
        if there is no test data, that some data is provided for
        evaluation. This function simply raises an error if the case
        is otherwise.

    Args:
        pretrained: boolean for whether or not the models are pretrained.
        X_train: Training data indepenent variables, default is None.
        y_train: Training data depenent variables, default is None.
        X_test: Test data indepenent variables, default is None. 
        y_test: Test data depenent variables, default is None. 

    Returns:
        None
    """
    if X_train is None or y_train is None:
        if not pretrained:
            raise ValueError("Either load saved models or provide training data.")
        if X_test is None and y_test is None:
            raise ValueError("Ensure either test or train data is provided.")


def build_and_evaluate(X_train=None, y_train=None, X_test=None, y_test=None, ext=None):
    """Builds and evaluates ML models.

    Args:
        X_train: Training data indepenent variables, default is None. If no
            complete test data is given, then X_train is used for evaluation.
        y_train: Training data depenent variables, default is None. If no
            complete test data is given, then y_train is used for evaluation.
        X_test: Test data indepenent variables, default is None. If None, then
            replaced by X_train for evaluation.
        y_test: Test data depenent variables, default is None. If None, then
            replaced by y_train for evaluation.
        ext: string extension of models file, default is None. If None, then
            untrained models are built. If not None, then the get_models function
            attempts to load the models object from the file f"models.{ext}".

    Returns:
        results: dict object where the keys are the model names, and the values
            are dict objects containing thr trained model, the classification
            report, and the roc_auc statistic.
    """
    models, pretrained = get_models(extension=ext)
    data_check(pretrained, X_train, y_train, X_test, y_test)
    results = {}
    hash_str = "#" * 60
    for model_name in models:
        outstr = hash_str + "\n" + model_name + "\n" + hash_str
        print(outstr)
        model = models[model_name]
        if not pretrained:
            print("Fitting model...")
            model.fit(X_train, y_train)
        if X_test is None or y_test is None:
            print("No testing data given. Evaluation will be done on training data.")
            X_test, y_test = X_train, y_train
        y_pred = model.predict(X_test)
        report = metrics.classification_report(y_test, y_pred, digits=3)
        print(f"Classification report:\n{report}")
        roc_auc = metrics.roc_auc_score(y_test, y_pred)
        print(f"Area under the ROC curve:\n{roc_auc:.3f}\n")
        results[model_name] = {
            "model": model,
            "classification_report": report,
            "roc_auc": roc_auc,
        }
    if not pretrained:
        save_models(models, ext)
    return results
