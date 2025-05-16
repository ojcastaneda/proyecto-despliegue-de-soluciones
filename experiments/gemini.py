from dotenv import load_dotenv
from humor_detection.llm import (
    predict,
    test_classification,
    test_detection,
    test_exclusive,
    test_lengths,
    test_repetition,
)
from humor_detection.utils import set_random_seeds
from pprint import pprint

load_dotenv()

# Cambiar semilla random
set_random_seeds()
model_name = "gemini-2.0-flash-lite"
# Prompts para predicciones
prompts = [
    "- Martínez, queda usted despedido.\n- Pero, si yo no he hecho nada.\n- Por eso, por eso.",  # Humor alto
    "¿Cuál es el último animal que subió al arca de Noé? El del-fin.",  # Humor
    "El otro día unas chicas llamarón a mi puerta y me pidieron una pequeña donación para una piscina local.\nLes di un garrafa de agua.",  # Humor
    "The brain surgeon changed my life. He really opened my mind.",  # No humor
    "djasndoasndoa",  # No humor
    "jajaja",  # No humor
]


def classification_prompter(input: str):
    return f"""You are a latino guy, rate the humor of the following text on a scale from 1 to 5, where 1 means not funny and 5 means very funny. Do not answer anything else.

Text: Un piano me caería excelente en estos momentos.
Output: 5

Text: Ni Jesús te ama.
Output: 1

Text: Jajajajajajajaj Idiota.
Output: 1

Text: {input}
Output: 
"""


def detection_prompter(input: str):
    return f"""You are a latino guy, rate the humor of the following text on a scale from 0 to 1, where 0 means not funny and 1 means funny. Do not answer anything else.

Text: Un piano me caería excelente en estos momentos.
Output: 1

Text: Ni Jesús te ama.
Output: 0

Text: Jajajajajajajaj Idiota.
Output: 0

Text: {input}
Output: 
"""


if __name__ == "__main__":
    pprint(test_classification(classification_prompter, model=model_name))
    # pprint(test_detection(detection_prompter))
    # pprint(test_exclusive(detection_prompter))
    # pprint(test_lengths(detection_prompter))
    # pprint(test_repetition(detection_prompter))
    # pprint(
    #     predict(prompts,
    #             ["1", "2", "3", "4", "5"],
    #             classification_prompter,
    #             model=model_name,
    #             threads=1)
    # )
    # pprint(
    #     predict(prompts,
    #             ["0", "1"],
    #             detection_prompter,
    #             model=model_name,
    #             threads=1)
    # )
