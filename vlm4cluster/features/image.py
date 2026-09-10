from __future__ import annotations

from pathlib import Path

from vlm4cluster.config import FeatureExtractorConfig, RuntimeConfig
from vlm4cluster.datasets.base import LoadedImageDataset, get_image_label
from vlm4cluster.features.base import FeatureSet, ImageFeatureExtractor
from vlm4cluster.features.registry import register_image_extractor
from vlm4cluster.utils.deps import require_module


def _sample_to_numpy(image):
    import numpy as np

    if hasattr(image, "detach"):
        return image.detach().cpu().numpy()
    if hasattr(image, "numpy"):
        return image.numpy()
    return np.asarray(image)


@register_image_extractor
class RawPixelsExtractor(ImageFeatureExtractor):
    name = "raw_pixels"
    description = "Flatten raw image pixels as clustering features."

    def extract(
        self,
        dataset: LoadedImageDataset,
        config: FeatureExtractorConfig,
        runtime: RuntimeConfig,
        output_dir: Path,
    ) -> FeatureSet:
        import numpy as np

        rows = []
        item_ids = []
        labels = [] if dataset.labels is not None else None

        for index in dataset.sample_indices:
            sample = dataset.dataset[index]
            image, raw_label = sample[0], sample[1]
            array = _sample_to_numpy(image).astype("float32")
            if array.max() > 1.0:
                array = array / 255.0
            rows.append(array.reshape(-1))
            item_ids.append(f"image-{index}")
            if labels is not None:
                labels.append(get_image_label(dataset, index, raw_label))

        matrix = np.stack(rows, axis=0)
        return FeatureSet(
            name=self.name,
            modality="image",
            matrix=matrix,
            labels=labels,
            item_ids=item_ids,
            metadata={"input_shape": list(rows[0].shape)},
        )


@register_image_extractor
class ResNet18ImageNetExtractor(ImageFeatureExtractor):
    name = "resnet18_imagenet"
    description = "Use torchvision ResNet-18 ImageNet features."

    def extract(
        self,
        dataset: LoadedImageDataset,
        config: FeatureExtractorConfig,
        runtime: RuntimeConfig,
        output_dir: Path,
    ) -> FeatureSet:
        import numpy as np

        torch = require_module("torch", "pip install torch torchvision")
        models = require_module("torchvision.models", "pip install torch torchvision")
        data_mod = require_module("torch.utils.data", "pip install torch torchvision")

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()

        device = torch.device(runtime.device)
        model.to(device)
        preprocess = weights.transforms()

        class EncodedSubset(data_mod.Dataset):
            def __init__(self, loaded_dataset, indices):
                self.loaded_dataset = loaded_dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                source_index = self.indices[item]
                image, raw_label = self.loaded_dataset.dataset[source_index]
                if hasattr(image, "detach"):
                    to_pil = require_module("torchvision.transforms", "pip install torch torchvision").ToPILImage()
                    image = to_pil(image)
                label = get_image_label(self.loaded_dataset, source_index, raw_label)
                return preprocess(image), label, f"image-{source_index}"

        view = EncodedSubset(dataset, dataset.sample_indices)
        loader = data_mod.DataLoader(
            view,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=runtime.num_workers,
        )

        features = []
        labels = []
        item_ids = []
        with torch.no_grad():
            for batch_images, batch_labels, batch_ids in loader:
                embeddings = model(batch_images.to(device))
                features.append(embeddings.cpu().numpy())
                labels.extend(int(label) for label in batch_labels.tolist())
                item_ids.extend(batch_ids)

        matrix = np.concatenate(features, axis=0)
        return FeatureSet(
            name=self.name,
            modality="image",
            matrix=matrix,
            labels=labels,
            item_ids=item_ids,
            metadata={"backbone": "resnet18", "weights": "imagenet1k_v1"},
        )
