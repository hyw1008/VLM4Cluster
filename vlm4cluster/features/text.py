from __future__ import annotations

from pathlib import Path

from vlm4cluster.config import FeatureExtractorConfig, RuntimeConfig
from vlm4cluster.datasets.base import LoadedTextDataset
from vlm4cluster.features.base import FeatureSet, TextFeatureExtractor
from vlm4cluster.features.registry import register_text_extractor


@register_text_extractor
class TfidfTextExtractor(TextFeatureExtractor):
    name = "tfidf"
    description = "TF-IDF text encoder for textual side information."

    def extract(
        self,
        dataset: LoadedTextDataset,
        config: FeatureExtractorConfig,
        runtime: RuntimeConfig,
        output_dir: Path,
    ) -> FeatureSet:
        from sklearn.feature_extraction.text import TfidfVectorizer

        max_features = int(config.params.get("max_features", 512))
        vectorizer = TfidfVectorizer(max_features=max_features)
        matrix = vectorizer.fit_transform(dataset.entries).toarray()
        return FeatureSet(
            name=self.name,
            modality="text",
            matrix=matrix,
            labels=dataset.labels,
            item_ids=dataset.item_ids,
            metadata={"max_features": max_features, "vocabulary_size": len(vectorizer.vocabulary_)},
        )
