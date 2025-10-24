from pathlib import Path
from typing import List

import pandas as pd
import pytest

from spam_email_detector.model import TextEncoder, Tfidf, Trainer, Word2Vec, predict

DATA_PATH = Path(__file__).parent.parent / "data" / "emails.csv"


@pytest.fixture(name = "train_data_path")
def fixture_train_data_path() -> pd.DataFrame:
    return DATA_PATH


@pytest.fixture(name = "test_emails")
def fixture_test_emails() -> List:
    return [
        "IMPORTANT: You won $500!!",
        "Hello Dan, I'm writing to let you know that we have a meeting next week and we'd want you to be present",
        "Congratulations! You’ve won a $1,000 gift card. Click here to claim your prize.",
        "Make money fast from home!!! Limited offer, sign up now.",
        "Urgent: your account has been suspended. Verify immediately by entering your password here.",
        "Your Amazon order #392817 has shipped.",
        "Meeting reminder: Project Sync at 3 PM",
        "October newsletter – new product updates and tips.",
    ]

@pytest.mark.slow
@pytest.mark.lfs_dependencies([str(DATA_PATH)])
@pytest.mark.parametrize("encoding_method", ["tf-idf", "word2vec"])
def test_integration(encoding_method: str, train_data_path: str, test_emails: list) -> None:
    # === Data ===
    df = pd.read_csv(train_data_path)

    # === Strategy ===
    if encoding_method == "tf-idf":
        strategy = Tfidf()
    elif encoding_method == "word2vec":
        strategy = Word2Vec()
    else:
        raise ValueError("Encoder type not supported")
    
    # === Encoder ===
    encoder = TextEncoder(strategy)

    # === Train Loop ===
    trainer = Trainer(encoder)
    model = trainer.train(df['text'], df["spam"].to_numpy())

    # === Inference ===
    for email in test_emails:
        pred = predict(email, encoder, model)
        print("Spam" if pred else "Not spam")