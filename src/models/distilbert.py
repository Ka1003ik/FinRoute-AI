"""Factory for the pretrained DistilBERT product classifier."""

from transformers import AutoModelForSequenceClassification

from src.features.bert_tokenizer import DISTILBERT_MODEL_ID, DISTILBERT_REVISION


def load_distilbert_classifier(labels, cache_dir):
    """Load the pinned pretrained encoder with a head for actual product labels."""
    id2label = {index: str(label) for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_ID,
        revision=DISTILBERT_REVISION,
        cache_dir=str(cache_dir),
        use_safetensors=True,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )
