"""
convert_openai_whisper_offline.py
====================================
Converts a local OpenAI-format Whisper .pt checkpoint (e.g. from
whisper.load_model(), cached at ~/.cache/whisper/large-v3.pt) into
HuggingFace transformers format, WITHOUT any network access.
 
Adapted directly from transformers' own conversion script
(src/transformers/models/whisper/convert_openai_to_hf.py) — same key
mapping, same config-building logic, verified against the real source.
The ONE piece removed is the call to GenerationConfig.from_pretrained(),
which fetches a small generation-behavior config from the HF Hub (mainly
used for word-level timestamp alignment heads) — that step needs
internet and is skipped here. The model will use transformers' own
built-in default generation config instead, which is fine for standard
transcription/fine-tuning; you only lose OpenAI's specific alignment-head
metadata for word-level timestamps, which this pipeline doesn't use.
 
Usage:
    python convert_openai_whisper_offline.py \
        --checkpoint_path ~/.cache/whisper/large-v3.pt \
        --output_dir ./whisper-large-v3-hf
 
Output: a directory with config.json + model.safetensors — weights only,
no tokenizer/processor files (see note printed at the end for those).
"""
 
import argparse
import torch
from torch import nn
from transformers import WhisperConfig, WhisperForConditionalGeneration
 
 
# Exact mapping from transformers' own convert_openai_to_hf.py —
# OpenAI's internal layer names -> HF's layer names.
WHISPER_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}
 
 
def remove_ignore_keys_(state_dict):
    for k in ("layers", "blocks"):
        state_dict.pop(k, None)
 
 
def rename_keys(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        new_key = key
        for old, new in WHISPER_MAPPING.items():
            if old in key:
                new_key = new_key.replace(old, new)
        if new_key != key:
            print(f"  {key}  ->  {new_key}")
        state_dict[new_key] = state_dict.pop(key)
    return state_dict
 
 
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer
 
 
def convert(checkpoint_path: str, output_dir: str):
    print(f"Loading local checkpoint: {checkpoint_path}")
    original_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
 
    dims = original_checkpoint["dims"]
    state_dict = original_checkpoint["model_state_dict"]
    proj_out_weights = state_dict["decoder.token_embedding.weight"]
 
    print("\nRenaming keys (OpenAI naming -> HF naming) ...")
    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
 
    ffn_dim = state_dict["decoder.layers.0.fc1.weight"].shape[0]
    endoftext_id = 50257 if dims["n_vocab"] > 51865 else 50256
 
    print("\nBuilding WhisperConfig from checkpoint dimensions ...")
    config = WhisperConfig(
        vocab_size=dims["n_vocab"],
        encoder_ffn_dim=ffn_dim,
        decoder_ffn_dim=ffn_dim,
        num_mel_bins=dims["n_mels"],
        d_model=dims["n_audio_state"],
        max_target_positions=dims["n_text_ctx"],
        encoder_layers=dims["n_audio_layer"],
        encoder_attention_heads=dims["n_audio_head"],
        decoder_layers=dims["n_text_layer"],
        decoder_attention_heads=dims["n_text_head"],
        max_source_positions=dims["n_audio_ctx"],
        eos_token_id=endoftext_id,
        bos_token_id=endoftext_id,
        pad_token_id=endoftext_id,
        decoder_start_token_id=endoftext_id + 1,
    )
 
    print("Instantiating HF model + loading converted weights ...")
    model = WhisperForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
 
    allowed_missing = {"encoder.embed_positions.weights", "decoder.embed_positions.weights"}
    if len(missing) > 0 and not set(missing) <= allowed_missing:
        raise ValueError(
            f"Unexpected missing weights (conversion likely broken): {missing}"
        )
    if unexpected:
        print(f"  NOTE: unexpected keys ignored during load: {unexpected}")
 
    # Tied embeddings — same as official script
    model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
 
    # NOTE: skipping GenerationConfig.from_pretrained(...) here — that call
    # needs internet access. Model falls back to transformers' built-in
    # default generation config. If you need OpenAI's specific alignment
    # heads for word-level timestamps later, that's the piece you'd be
    # missing — not needed for standard fine-tuning/transcription.
 
    print(f"\nSaving HF-format model to {output_dir} ...")
    model.save_pretrained(output_dir)
    print("Done — weights + config.json written.")
    print(
        "\nNOTE: this only writes model weights, no tokenizer/processor files.\n"
        "The tokenizer/feature-extractor for whisper-large-v3 are small\n"
        "(a few MB total) and IDENTICAL regardless of fine-tuning — they\n"
        "don't need conversion. Grab them once from any machine with\n"
        "internet access, e.g. from that machine's own HF cache after running\n"
        "WhisperProcessor.from_pretrained('openai/whisper-large-v3'):\n"
        "  ~/.cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/*/\n"
        "    (tokenizer_config.json, tokenizer.json, vocab.json, merges.txt,\n"
        "     preprocessor_config.json)\n"
        f"Copy those files into {output_dir}/ alongside config.json, and\n"
        "WhisperProcessor.from_pretrained(output_dir) will work fully offline."
    )
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to the local .pt file, e.g. ~/.cache/whisper/large-v3.pt")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the converted HF-format model")
    args = parser.parse_args()
 
    convert(args.checkpoint_path, args.output_dir)
