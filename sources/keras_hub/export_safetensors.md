# KerasHub models exportable to SafeTensors

This page lists all KerasHub models that can be exported to HuggingFace SafeTensors format. SafeTensors is a safe serialization format for storing and loading tensors, making it ideal for sharing and distributing pretrained models.

## Using Export to SafeTensors

You can export any of the following models to HuggingFace Transformers format (SafeTensors) using the `export_to_transformers()` method:

```python
import keras_hub

# Load a KerasHub model
gemma_model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
tokenizer = keras_hub.models.GemmaTokenizer.from_preset("gemma_2b_en")
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor(tokenizer=tokenizer)

# Create the full model
causal_lm = keras_hub.models.GemmaCausalLM(backbone=gemma_model.backbone, preprocessor=preprocessor)

# Export to HuggingFace format (creates safetensors files)
causal_lm.export_to_transformers("path/to/export_dir")

# Load in HuggingFace Transformers
from transformers import AutoModel, AutoTokenizer
hf_model = AutoModel.from_pretrained("path/to/export_dir")
hf_tokenizer = AutoTokenizer.from_pretrained("path/to/export_dir")
```

## Exportable Models

{{safetensors_table}}

## Export Features

All models listed above support the following export features:

- **Backbone Export**: Export just the model backbone architecture and weights in HuggingFace format
- **Tokenizer Export**: Export the tokenizer configuration and vocabulary files in HuggingFace format
- **Full Model Export**: Export the complete model with backbone, tokenizer, and preprocessing layers
- **SafeTensors Format**: Models are automatically saved in the safe and efficient SafeTensors format
- **HuggingFace Compatible**: Exported models are fully compatible with HuggingFace Transformers library

## Export to Different Formats

For additional export options, see the KerasHub models API documentation:

- Export to **PyTorch format** via tools
- Export to **TFLite format** for mobile inference
- Export to **SavedModel format** for TensorFlow serving

## Related Resources

- [KerasHub Getting Started Guide](/keras_hub/getting_started/)
- [KerasHub Pretrained Models](/keras_hub/presets/)
- [HuggingFace Transformers Integration](/keras_hub/guides/hugging_face_keras_integration/)
