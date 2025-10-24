import argparse
import logging
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple, Union

import nltk
import numpy as np
import pandas as pd
import torch
from gensim.downloader import load as gensim_load
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__}")
print(f"DEVICE={DEVICE}")

# Determinism
SEED: int = 317
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Abstract base class — explicit interface
class EncodingStrategy(ABC):
    
    def __init__(self) -> None:
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> List[str]:
        """
        Clean and preprocess a text string.

        Args:
            text: The raw text to clean.

        Returns:
            List of lemmatized tokens with punctuation and stopwords removed.
        """
        words = [word for word in text.split()]
        words = [re.sub(r"[^a-zA-Z0-9']", "", word) for word in words]
        words = [word for word in words if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in words]
        return tokens

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def transform(self, text: str) -> np.ndarray:
        pass


# Concrete strategies
class Tfidf(EncodingStrategy):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Building TF-IDF encoder...")
        self.vectorizer = TfidfVectorizer(tokenizer=self.clean_text)
        logger.info("Encoder built successfully.")

    def encode(self, text: Union[List[str], pd.Series]) -> np.ndarray:
        logger.info("Encoding text corpus...")
        output = self.vectorizer.fit_transform(text).toarray().astype(np.float32)
        logger.info("Encoding successful")
        return output
    
    def transform(self, text: str) -> np.ndarray:
        output = self.vectorizer.transform([text]).toarray().astype(np.float32)
        return output


class Word2Vec(EncodingStrategy):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Building word2vec encoder...")
        self.w2v_encoder = gensim_load("glove-wiki-gigaword-100")
        logger.info("Encoder built successfully.")
    
    def encode_single(self, text: str) -> np.ndarray:
        words = self.clean_text(text)
        words = [w for w in words if w in self.w2v_encoder.key_to_index]
        if not words:
            return np.zeros(self.w2v_encoder.vector_size, dtype=np.float32)
        return np.mean(self.w2v_encoder(words), axis=0)
    
    def encode(self, text: str) -> np.ndarray:
        logger.info("Encoding text corpus...")
        encodings = []
        for _, t in tqdm(enumerate(text)):
            encodings.append(self.encode_single(t))
        logger.info("Encoding successful")
        output = np.array(encodings, dtype=np.float32)
        output = output[None, :]
        return output

    # TODO: transform method


class TextEncoder():

    def __init__(self, strategy: EncodingStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: EncodingStrategy) -> None:
        self._strategy = strategy

    def encode(self, text: str) -> np.ndarray:
        return self._strategy.encode(text)
    
    def transform(self, text: str) -> np.ndarray:
        return self._strategy.transform(text)


class LinearTextClassifier(nn.Module):
    """A simple linear text classifier using one fully connected layer."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        """
        Initialize the linear classifier.

        Args:
            input_dim: The number of input features.
            num_classes: Number of output classes. Defaults to 2.
        """
        super().__init__()
        self.layer = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after the linear layer.
        """
        return self.layer(x)


class Trainer:

    def __init__(self, encoder: TextEncoder, epochs: int = 200, lr: float = 0.01) -> None:
        self.encoder = encoder
        self.epochs = epochs
        self.lr = lr

    def train(self, text: Union[List[str], pd.DataFrame], y: np.ndarray, do_eval: bool = True) -> nn.Module:
        X = self.encoder.encode(text)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
        
        model = LinearTextClassifier(len(X_train[0]))
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=self.lr)

        X_train_t = torch.tensor(X_train)
        X_test_t = torch.tensor(X_test)
        y_train_t = torch.tensor(y_train)
        y_test_t = torch.tensor(y_test)

        # === Evaluation ===
        if do_eval:
            for epoch in tqdm(range(self.epochs)):
                model.train()
                optimizer.zero_grad()
                logits = model(X_train_t)
                loss = criterion(logits, y_train_t)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0:
                    with torch.no_grad():
                        model.eval()
                        pred = model(X_test_t)
                        pred_labels = torch.argmax(pred, -1)

                        tp = ((pred_labels == 1) & (y_test_t == 1)).sum().item()
                        tn = ((pred_labels == 0) & (y_test_t == 0)).sum().item()
                        fp = ((pred_labels == 1) & (y_test_t == 0)).sum().item()
                        fn = ((pred_labels == 0) & (y_test_t == 1)).sum().item()

                        acc = (tp + tn) / (tp + tn + fp + fn)
                        test_loss = criterion(pred, y_test_t)

                        print(f"Epoch {epoch+1}: Accuracy={acc:.3f}, Test loss={test_loss.item():.4f}")

        return model


def predict(text: str, encoder: TextEncoder, model: LinearTextClassifier) -> torch.Tensor:
    """
    Predict whether a message is spam or not.

    Args:
        msg: The input message text.
        encoder: The trained encoder (TF-IDF vectorizer or Word2Vec model).
        method_to_use: Encoding method ('tf-idf' or 'word2vec').
        model: Trained classifier model.

    Returns:
        Tensor containing the predicted label.
    """
    input_enc = encoder.transform(text)
    input_t = torch.tensor(input_enc)
    logits = model(input_t)
    pred = bool(torch.argmax(logits, dim=1).item())
    return pred


if __name__ == "__main__":
    # === Arg Parsing ===
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", choices=["tf-idf", "word2vec"])
    args = parser.parse_args()

    # === Data ===
    df = pd.read_csv(Path(__file__).parent / "emails.csv")

    # === Strategy ===
    if args.encoder == "tf-idf":
        strategy = Tfidf()
    elif args.encoder == "word2vec":
        strategy = Word2Vec()
    else:
        raise ValueError("Encoder type not supported")
    
    # === Encoder ===
    encoder = TextEncoder(strategy)

    # === Train Loop ===
    trainer = Trainer(encoder)
    model = trainer.train(df['text'], df["spam"].to_numpy())
