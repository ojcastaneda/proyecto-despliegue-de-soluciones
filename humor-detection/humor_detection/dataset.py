from abc import ABC, abstractmethod
from json import load
from os import makedirs
from os.path import exists, dirname
from typing import Callable
from pandas import DataFrame, concat, read_csv
from ydata_profiling import ProfileReport

DATA_FOLDER = "../data"

CLASSIFICATION_FOLDER = f"{DATA_FOLDER}/classification"
DETECTION_FOLDER = f"{DATA_FOLDER}/detection"
PROCESSED_FOLDER = f"{DATA_FOLDER}/processed"
SYNTHETIC_FOLDER = f"{DATA_FOLDER}/synthetic"
RAW_FOLDER = f"{DATA_FOLDER}/raw"
TEST_PATH = f"{DATA_FOLDER}/test.csv"


class DatasetProcessor(ABC):
    output_path: str

    @abstractmethod
    def preprocess(self) -> DataFrame:
        pass

    def process(
        self,
        output_path: str | None = None,
        preprocess: Callable[[], DataFrame] | None = None,
        report=True,
    ):
        if output_path is None:
            output_path = self.output_path
        makedirs(dirname(output_path), exist_ok=True)
        if preprocess is None:
            preprocess = self.preprocess
        if not exists(output_path):
            dataset = preprocess()
            dataset.to_csv(output_path, index=False)
            return ProfileReport(dataset) if report else None
        return ProfileReport(load_csv(output_path)) if report else None


class FinalDatasetProcessor(DatasetProcessor):
    classification_path: str
    detection_path: str

    @abstractmethod
    def preprocess_classification(self) -> DataFrame:
        pass

    @abstractmethod
    def preprocess_detection(self) -> DataFrame:
        pass

    def process(self, report=True):
        base = super().process(report=report)
        classification = super().process(
            self.classification_path, self.preprocess_classification, report
        )
        detection = super().process(
            self.detection_path, self.preprocess_detection, report
        )
        return base, classification, detection


class Test(FinalDatasetProcessor):
    output_path = f"{DATA_FOLDER}/test_no_labels.csv"
    classification_path = f"{CLASSIFICATION_FOLDER}/test.csv"
    detection_path = f"{DETECTION_FOLDER}/test.csv"

    def preprocess(self):
        return concat(
            [load_csv(SpanishJokes), load_csv(HAHATest)],
            ignore_index=True,
        ).drop_duplicates(subset=["text"], keep=False)

    def preprocess_classification(self):
        dataset = load_csv(TEST_PATH)
        id_counts = dataset["id"].value_counts()
        dataset = dataset[dataset["id"].isin(id_counts[id_counts >= 4].index)]

        def get_majority_sentiment(x):
            non_zeros = x[x != 0]
            if len(non_zeros) == 0:
                return 0
            value_counts = non_zeros.value_counts()
            max_count = value_counts.max()
            return (
                value_counts[value_counts == max_count].index[0]
                if (max_count / len(non_zeros)) > 0.5
                else non_zeros.max()
            )

        dataset = dataset.groupby("id").agg(
            text=("text", "first"),
            zero_count=("sentiment", lambda x: (x == 0).sum()),
            non_zero_count=("sentiment", lambda x: (x != 0).sum()),
            majority_sentiment=("sentiment", get_majority_sentiment),
        )
        dataset = dataset[dataset["non_zero_count"] >= dataset["zero_count"]]
        dataset["score"] = (dataset["majority_sentiment"] - 1).astype(int)
        return dataset[["text", "score"]]

    def preprocess_detection(self):
        dataset = load_csv(TEST_PATH)
        id_counts = dataset["id"].value_counts()
        dataset = dataset[dataset["id"].isin(id_counts[id_counts >= 4].index)]
        dataset = dataset.groupby("id").agg(
            text=("text", "first"),
            zero_count=("sentiment", lambda x: (x == 0).sum()),
            non_zero_count=("sentiment", lambda x: (x != 0).sum()),
        )
        dataset["score"] = (dataset["zero_count"] <= dataset["non_zero_count"]).astype(
            int
        )
        return dataset[["text", "score"]]


class TestExclusive(DatasetProcessor):
    output_path = f"{SYNTHETIC_FOLDER}/exclusive.csv"


class TestLongLengths(DatasetProcessor):
    output_path = f"{SYNTHETIC_FOLDER}/long_lengths.csv"


class TestRepetition(DatasetProcessor):
    output_path = f"{SYNTHETIC_FOLDER}/repetition.csv"


class TestShortLengths(DatasetProcessor):
    output_path = f"{SYNTHETIC_FOLDER}/short_lengths.csv"


class Train(FinalDatasetProcessor):
    output_path = f"{DATA_FOLDER}/train.csv"
    classification_path = f"{CLASSIFICATION_FOLDER}/train.csv"
    detection_path = f"{DETECTION_FOLDER}/train.csv"

    def preprocess(self):
        return concat(
            [load_csv(HAHATrain), load_csv(f"{DATA_FOLDER}/generated_jokes.csv")],
            ignore_index=True,
        ).drop_duplicates(subset=["text"], keep=False)

    def preprocess_classification(self):
        return self.transform_classification(load_csv(Train))

    def preprocess_detection(self):
        return self.transform_detection(load_csv(Train))

    @staticmethod
    def transform_classification(dataset: DataFrame):
        dataset = dataset[dataset["score"] > 0]
        dataset.loc[:, "score"] = dataset["score"] - 1
        return dataset

    @staticmethod
    def transform_detection(dataset: DataFrame):
        dataset["score"] = (dataset["score"] != 0).astype(int)
        return dataset


class TrainMultilingual(FinalDatasetProcessor):
    output_path = f"{DATA_FOLDER}/train_multilingual.csv"
    classification_path = f"{CLASSIFICATION_FOLDER}/train_multilingual.csv"
    detection_path = f"{DETECTION_FOLDER}/train_multilingual.csv"

    def preprocess(self):
        return concat(
            [load_csv(HAHATrain), load_csv(StupidStuff)],
            ignore_index=True,
        ).drop_duplicates(subset=["text"], keep=False)

    def preprocess_classification(self):
        return Train.transform_classification(load_csv(TrainMultilingual))

    def preprocess_detection(self):
        return Train.transform_detection(load_csv(TrainMultilingual))


class HAHATest(DatasetProcessor):
    output_path = f"{PROCESSED_FOLDER}/haha_test.csv"

    def preprocess(self):
        return load_csv(f"{RAW_FOLDER}/haha_test.csv")[["text"]]


class HAHATrain(DatasetProcessor):
    output_path = f"{PROCESSED_FOLDER}/haha_train.csv"

    def preprocess(self):
        dataset = load_csv(f"{RAW_FOLDER}/haha_train.csv")
        result = DataFrame()
        result["text"] = dataset["text"]
        result["score"] = 0
        for idx in dataset[dataset["is_humor"] != 0].index:
            votes = []
            total_votes = 0
            for i in range(1, 6):
                vote_count = int(dataset.loc[idx, f"votes_{i}"])  # type: ignore
                votes.append(vote_count)
                total_votes += vote_count
            majority_found = False
            for i, vote_count in enumerate(votes, 1):
                if vote_count / total_votes > 0.5:
                    result.loc[idx, "score"] = i
                    majority_found = True
                    break
            if not majority_found:
                for i in range(5, 0, -1):
                    if (votes[i - 1] / total_votes) > 0.2:
                        result.loc[idx, "score"] = i
                        break
        result["score"] = result["score"].clip(0, 5)
        return result


class SpanishJokes(DatasetProcessor):
    output_path = f"{PROCESSED_FOLDER}/spanish_jokes.csv"

    def preprocess(self):
        dataset = load_csv(f"{RAW_FOLDER}/spanish_jokes.csv").drop_duplicates(
            subset=["text"]
        )
        return DataFrame([{"text": item["text"]} for _, item in dataset.iterrows()])


class StupidStuff(DatasetProcessor):
    output_path = f"{PROCESSED_FOLDER}/stupid_stuff.csv"

    def preprocess(self) -> DataFrame:
        with open(f"{RAW_FOLDER}/stupid_stuff.json", "r") as file:
            dataset = load(file)
        unique_items = {}
        to_remove = set([""])
        for item in dataset:
            if (
                item["body"] in unique_items
                and unique_items[item["body"]] != item["rating"]
            ):
                to_remove.add(item["body"])
            unique_items[item["body"]] = item["rating"]
        return DataFrame(
            [
                {"text": item["body"], "score": round(item["rating"])}
                for item in dataset
                if item["body"] not in to_remove and len(item["body"]) <= 512
            ]
        ).drop_duplicates(subset=["text"], keep="first")


def load_csv(path: str | type[DatasetProcessor]):
    return read_csv(
        path if isinstance(path, str) else path.output_path, encoding="utf-8"
    )
