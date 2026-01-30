"""Automatic rendering for special KerasHub preset collections.

Generates markdown tables for curated model collections by querying
KerasHub presets and categorizing them by model type.
"""

try:
    import keras_hub
except Exception as e:
    print(f"Could not import KerasHub. Exception: {e}")
    keras_hub = None


def get_safetensors_models():
    """Get models that support SafeTensors export.
    
    Discovers from KerasHub CausalLM presets by checking model family names.
    """
    if keras_hub is None:
        return {}
    
    safetensors_models = {}
    try:
        if hasattr(keras_hub.models.CausalLM, 'presets'):
            for preset_name in keras_hub.models.CausalLM.presets.keys():
                family = None
                if preset_name.startswith("gemma3"):
                    family = "Gemma3"
                elif preset_name.startswith("gemma2"):
                    family = "Gemma2"
                elif preset_name.startswith("gemma"):
                    family = "Gemma"
                elif preset_name.startswith("qwen3"):
                    family = "Qwen3"
                elif preset_name.startswith("qwen"):
                    family = "Qwen"
                
                if family:
                    if family not in safetensors_models:
                        safetensors_models[family] = []
                    safetensors_models[family].append(preset_name)
    except Exception as e:
        print(f"Error discovering SafeTensors models: {e}")
    
    return safetensors_models


def get_huggingface_models():
    """Get models that support HuggingFace conversion.
    
    Discovers from KerasHub presets and categorizes by model type.
    """
    if keras_hub is None:
        return {}
    
    huggingface_models = {
        "Text Encoders": [],
        "Text Generation": [],
        "Vision": [],
        "Seq2Seq": [],
        "Other": [],
    }
    
    # Map model types to categories
    text_encoder_types = ["albert", "bert", "distilbert", "electra", "roberta", "xlm", "f_net", "esm"]
    vision_types = ["vit", "deit", "dinov2", "dinov3_vit"]
    text_gen_types = ["gemma", "gemma2", "gemma3", "gpt2", "llama", "mistral", "qwen2", "qwen3", "paligemma"]
    seq2seq_types = ["bart", "t5"]
    
    try:
        # Text encoders and vision from Backbone
        if hasattr(keras_hub.models.Backbone, 'presets'):
            for preset_name in keras_hub.models.Backbone.presets.keys():
                added = False
                for enc_type in text_encoder_types:
                    if enc_type in preset_name:
                        huggingface_models["Text Encoders"].append(preset_name)
                        added = True
                        break
                if not added:
                    for vis_type in vision_types:
                        if vis_type in preset_name:
                            huggingface_models["Vision"].append(preset_name)
                            break
        
        # Text generation from CausalLM
        if hasattr(keras_hub.models.CausalLM, 'presets'):
            for preset_name in keras_hub.models.CausalLM.presets.keys():
                huggingface_models["Text Generation"].append(preset_name)
        
        # Seq2Seq models
        if hasattr(keras_hub.models.Seq2SeqLM, 'presets'):
            for preset_name in keras_hub.models.Seq2SeqLM.presets.keys():
                huggingface_models["Seq2Seq"].append(preset_name)
    except Exception as e:
        print(f"Error discovering HuggingFace models: {e}")
    
    return huggingface_models


# Auto-generate model lists at module load time
SAFETENSORS_MODELS = get_safetensors_models()
HUGGINGFACE_MODELS = get_huggingface_models()




def format_param_count(metadata):
    """Format a parameter count for the table."""
    try:
        count = metadata["params"]
    except KeyError:
        return "Unknown"
    if count >= 1e9:
        return f"{(count / 1e9):.2f}B"
    if count >= 1e6:
        return f"{(count / 1e6):.2f}M"
    if count >= 1e3:
        return f"{(count / 1e3):.2f}K"
    return f"{count}"


def get_preset_data(preset_name):
    """Get metadata for a preset from KerasHub."""
    if keras_hub is None:
        return None
    
    # Search across all model types
    for model_class in [keras_hub.models.CausalLM, keras_hub.models.Backbone]:
        try:
            if hasattr(model_class, 'presets') and preset_name in model_class.presets:
                return model_class.presets[preset_name]
        except Exception:
            pass
    
    return None


def render_model_family_table(family_name, preset_names):
    """Render a markdown table row for a model family."""
    rows = []
    for preset_name in preset_names:
        data = get_preset_data(preset_name)
        if data is None:
            continue
        
        metadata = data.get("metadata", {})
        params = format_param_count(metadata)
        description = metadata.get("description", "")
        
        row = f"| **{family_name}** | `{preset_name}` | {params} | {description} |"
        rows.append(row)
        family_name = ""  # Only show family name for first row
    
    return "\n".join(rows)


def render_safetensors_table():
    """Generate markdown table for SafeTensors-exportable models."""
    if keras_hub is None:
        return "Unable to load model data. Ensure KerasHub is installed."
    
    table = "| Model Family | Preset Name | Parameters | Description |\n"
    table += "|---|---|---|---|\n"
    
    for family, presets in SAFETENSORS_MODELS.items():
        family_rows = render_model_family_table(family, presets)
        if family_rows:
            table += family_rows + "\n"
    
    return table


def render_huggingface_table():
    """Generate markdown table for HuggingFace-convertible models."""
    if keras_hub is None:
        return "Unable to load model data. Ensure KerasHub is installed."
    
    sections = []
    
    for section_name, presets in HUGGINGFACE_MODELS.items():
        section = f"\n### {section_name}\n\n"
        section += "| Model Family | Preset Name | Description | Parameters |\n"
        section += "|---|---|---|---|\n"
        
        for preset_name in presets:
            data = get_preset_data(preset_name)
            if data is None:
                continue
            
            metadata = data.get("metadata", {})
            params = format_param_count(metadata)
            description = metadata.get("description", "")
            
            # Extract family name from preset
            family = preset_name.split("_")[0].upper()
            
            row = f"| **{family}** | `{preset_name}` | {description} | {params} |"
            section += row + "\n"
        
        sections.append(section)
    
    return "\n".join(sections)


def render_tags(template):
    """Replace custom tags in template with generated content."""
    if keras_hub is None:
        return template
    
    if "{{safetensors_table}}" in template:
        table = render_safetensors_table()
        template = template.replace("{{safetensors_table}}", table)
    
    if "{{huggingface_table}}" in template:
        table = render_huggingface_table()
        template = template.replace("{{huggingface_table}}", table)
    
    return template
