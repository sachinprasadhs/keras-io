# KerasHub models convertible from HuggingFace

This page lists all KerasHub models that support on-the-fly conversion from HuggingFace Hub. This includes both HuggingFace Transformers models and TIMM (PyTorch Image Models) vision models. These models can be loaded directly from HuggingFace Hub using the `hf://` protocol, with automatic weight conversion to the KerasHub format.

## Loading Models from HuggingFace

KerasHub supports loading models directly from HuggingFace Hub with automatic conversion using the `hf://` protocol:

```python
import keras_hub

# Load text models from HuggingFace Hub
backbone = keras_hub.models.Backbone.from_preset("hf://bert-base-uncased")
tokenizer = keras_hub.models.BertTokenizer.from_preset("hf://bert-base-uncased")

# Load full classification model
classifier = keras_hub.models.TextClassifier.from_preset(
    "hf://bert-base-uncased",
    num_classes=2
)

# Load causal language models
lm = keras_hub.models.CausalLM.from_preset("hf://mistralai/Mistral-7B-v0.1")

# Load TIMM vision models from HuggingFace Hub
vision_backbone = keras_hub.models.EfficientNetBackbone.from_preset(
    "hf://timm/efficientnet_b0.in1k"
)

# Create image classifier
image_classifier = keras_hub.models.ImageClassifier.from_preset(
    "hf://timm/mobilenetv3_large_100.ra_in1k",
    num_classes=1000
)
```

## Convertible Model Families

The following model families support automatic conversion from HuggingFace Hub, including both Transformers models and TIMM vision models.

{{huggingface_table}}

## On-the-Fly Conversion Process

When loading a model from HuggingFace, KerasHub performs the following steps automatically:

1. **Download Model**: Fetches the model config and weights from HuggingFace Hub
2. **Convert Architecture**: Translates the HuggingFace model configuration to KerasHub format
3. **Port Weights**: Converts and remaps weights from HuggingFace to KerasHub layer structure
4. **Initialize Tokenizer**: Extracts and initializes the tokenizer from HuggingFace vocabulary (for text models)

## Benefits of HuggingFace Integration

- **Direct Access**: Load any compatible HuggingFace model directly without manual conversion
- **Automatic Conversion**: Weights and tokenizers are automatically converted on-the-fly
- **Memory Efficient**: Models are converted and cached for subsequent loads
- **Easy Fine-tuning**: Train and fine-tune HuggingFace models using KerasHub's unified API
- **Load Fine-tuned Models**: Load your own or community fine-tuned models from HuggingFace Hub using the same `hf://` protocol
- **Export Support**: Convert back to HuggingFace SafeTensors format when needed
- **TIMM Vision Models**: Access to hundreds of pre-trained vision models from the TIMM library via HuggingFace Hub

## Limitations and Notes

- Some HuggingFace models may have architectural differences that require custom adaptation
- Floating-point precision conversions may introduce small numerical differences
- Very large models may require significant memory during conversion
- For best results, use models with clear HuggingFace documentation
- TIMM models are optimized for vision tasks and may require specific preprocessing

## Related Resources

- [KerasHub Getting Started Guide](/keras_hub/getting_started/)
- [KerasHub API Reference](/keras_hub/api/models/)
- [HuggingFace Keras Integration Guide](/keras_hub/guides/hugging_face_keras_integration/)
- [Classification with KerasHub](/keras_hub/guides/classification_with_keras_hub/)
- [Models Exportable to SafeTensors](/keras_hub/export_safetensors/)
