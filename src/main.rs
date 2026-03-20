mod frontend;
mod model;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::qwen2::{Config as Qwen2Config, Model as Qwen2Model};
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use tokenizers::Tokenizer;
use std::path::PathBuf;

use crate::frontend::WavFrontend;
use crate::model::{SenseVoiceEncoderSmall, AudioAdaptor, CTCDecoder};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    tokenizer: PathBuf,

    #[arg(long)]
    wav: PathBuf,

    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        "metal" => Device::new_metal(0)?,
        _ => return Err(anyhow!("Invalid device")),
    };

    // 1. Load Configs
    let config_str = std::fs::read_to_string(&args.config)?;
    let config: serde_yaml::Value = serde_yaml::from_str(&config_str)?;

    // 2. Load Model Weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.model], DType::F32, &device)?
    };

    // 3. Initialize Components
    let frontend_conf = &config["frontend_conf"];
    let frontend = WavFrontend::new(
        frontend_conf["fs"].as_u64().unwrap_or(16000) as usize,
        frontend_conf["n_mels"].as_u64().unwrap_or(80) as usize,
        frontend_conf["frame_length"].as_u64().unwrap_or(25) as usize,
        frontend_conf["frame_shift"].as_u64().unwrap_or(10) as usize,
        frontend_conf["lfr_m"].as_u64().unwrap_or(7) as usize,
        frontend_conf["lfr_n"].as_u64().unwrap_or(6) as usize,
    );

    let enc_conf = &config["audio_encoder_conf"];
    let encoder = SenseVoiceEncoderSmall::load(
        vb.pp("audio_encoder"),
        frontend_conf["n_mels"].as_u64().unwrap_or(80) as usize,
        enc_conf["output_size"].as_u64().unwrap_or(512) as usize,
        enc_conf["attention_heads"].as_u64().unwrap_or(4) as usize,
        enc_conf["linear_units"].as_u64().unwrap_or(2048) as usize,
        enc_conf["num_blocks"].as_u64().unwrap_or(50) as usize,
        enc_conf["tp_blocks"].as_u64().unwrap_or(20) as usize,
        enc_conf["kernel_size"].as_u64().unwrap_or(11) as usize,
        enc_conf["sanm_shift"].as_u64().unwrap_or(0) as usize,
    )?;

    let adaptor_conf = &config["audio_adaptor_conf"];
    let adaptor = AudioAdaptor::load(
        vb.pp("audio_adaptor"),
        adaptor_conf["encoder_dim"].as_u64().unwrap_or(512) as usize,
        adaptor_conf["llm_dim"].as_u64().unwrap_or(1024) as usize,
        adaptor_conf["n_layer"].as_u64().unwrap_or(2) as usize,
    )?;

    let ctc_conf = &config["ctc_decoder_conf"];
    let ctc_decoder = CTCDecoder::load(
        vb.pp("ctc_decoder"),
        ctc_conf["encoder_dim"].as_u64().unwrap_or(512) as usize,
        ctc_conf["llm_dim"].as_u64().unwrap_or(512) as usize,
        ctc_conf["n_layer"].as_u64().unwrap_or(5) as usize,
        60515, // Default vocab size
        60514, // Default blank id
    )?;

    // Qwen3 (similar to Qwen2)
    let qwen_config_path = args.config.parent().unwrap().join("Qwen3-0.6B").join("config.json");
    let qwen_config_str = std::fs::read_to_string(qwen_config_path)?;
    let qwen_config: Qwen2Config = serde_json::from_str(&qwen_config_str)?;

    let qwen_emb = candle_nn::embedding(qwen_config.vocab_size, qwen_config.hidden_size, vb.pp("llm.model.embed_tokens"))?;
    let mut qwen = Qwen2Model::new(&qwen_config, vb.pp("llm"))?;

    let tokenizer = Tokenizer::from_file(args.tokenizer).map_err(|e| anyhow!(e))?;

    // 4. Audio Preprocessing
    let reader = hound::WavReader::open(args.wav)?;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let fbank = frontend.extract_fbank(&samples)?;
    let (lfr_feat, t_lfr) = frontend.apply_lfr(&fbank);
    let feat_tensor = Tensor::from_vec(lfr_feat, (1, t_lfr, 80 * 7), &device)?;
    let ilens = Tensor::from_vec(vec![t_lfr as u32], (1,), &device)?;

    // 5. Encoder and Adaptor forward
    let (enc_out, _olens) = encoder.forward(&feat_tensor, &ilens)?;
    let (adaptor_out, _olens) = adaptor.forward(&enc_out, &ilens)?;

    // 6. CTC Path (for timestamps)
    let ctc_logits = ctc_decoder.forward(&enc_out, &ilens)?;
    let ctc_ids = ctc_logits.squeeze(0)?.argmax(1)?;
    let ctc_ids_vec = ctc_ids.to_vec1::<i64>()?;

    // Simple CTC Greedy Decoding for reference text (though we usually use LLM text)
    let mut prev_id = -1;
    let mut decoded_ctc_ids = Vec::new();
    for &id in ctc_ids_vec.iter() {
        if id != 60514 && id != prev_id {
            decoded_ctc_ids.push(id as u32);
        }
        prev_id = id;
    }
    let ctc_text = tokenizer.decode(&decoded_ctc_ids, true).map_err(|e| anyhow!(e))?;
    println!("CTC Result: {}", ctc_text);

    // 7. LLM Integration
    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：<|startofspeech|>!!<|endofspeech|><|im_end|>\n<|im_start|>assistant\n";
    let parts: Vec<&str> = prompt.split("!!").collect();
    let tokens_before = tokenizer.encode(parts[0], true).map_err(|e| anyhow!(e))?;
    let tokens_after = tokenizer.encode(parts[1], true).map_err(|e| anyhow!(e))?;

    let ids_before = Tensor::new(tokens_before.get_ids(), &device)?.unsqueeze(0)?;
    let ids_after = Tensor::new(tokens_after.get_ids(), &device)?.unsqueeze(0)?;

    let embed_before = qwen_emb.forward(&ids_before)?;
    let embed_after = qwen_emb.forward(&ids_after)?;

    // Concatenate: [embed_before, adaptor_out, embed_after]
    let inputs_embeds = Tensor::cat(&[&embed_before, &adaptor_out, &embed_after], 1)?;

    // 8. Greedy Decoding (LLM)
    let mut logits_processor = LogitsProcessor::new(42, None, None);
    let mut generated_tokens = Vec::new();
    let mut current_embeds = inputs_embeds;

    let eos_token_id = tokenizer.token_to_id("<|im_end|>").or_else(|| tokenizer.token_to_id("<|endoftext|>")).unwrap_or(151643);

    for _i in 0..512 {
        let logits = qwen.forward(&current_embeds, 0, None)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let next_token = logits_processor.sample(&logits)?;

        if next_token == eos_token_id {
            break;
        }

        generated_tokens.push(next_token);

        let next_id = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let next_emb = qwen_emb.forward(&next_id)?;
        current_embeds = Tensor::cat(&[&current_embeds, &next_emb], 1)?;
    }

    let decoded = tokenizer.decode(&generated_tokens, true).map_err(|e| anyhow!(e))?;
    println!("ASR Result: {}", decoded);

    // 9. Timestamps (Forced Alignment placeholder)
    // In a full implementation, we'd call ctc_forced_align(ctc_logits, decoded_ids, ...)
    // and then convert frame indices to time.
    // Frame duration: 10ms (frame_shift) * 6 (lfr_n) = 60ms per LFR frame.

    Ok(())
}
