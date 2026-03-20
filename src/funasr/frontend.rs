use anyhow::{Context, Result};
use rustfft::{num_complex::Complex, FftPlanner};

pub struct WavFrontend {
    fs: usize,
    n_mels: usize,
    frame_length_ms: usize,
    frame_shift_ms: usize,
    lfr_m: usize,
    lfr_n: usize,
    mel_filters: Vec<Vec<f32>>,
    window: Vec<f32>,
}

impl WavFrontend {
    pub fn new(
        fs: usize,
        n_mels: usize,
        frame_length_ms: usize,
        frame_shift_ms: usize,
        lfr_m: usize,
        lfr_n: usize,
    ) -> Self {
        let frame_length = fs * frame_length_ms / 1000;
        let window = hamming_window(frame_length);
        let mel_filters = create_mel_filters(n_mels, frame_length, fs as f32);

        Self {
            fs,
            n_mels,
            frame_length_ms,
            frame_shift_ms,
            lfr_m,
            lfr_n,
            mel_filters,
            window,
        }
    }

    pub fn extract_fbank(&self, waveform: &[f32]) -> Result<Vec<Vec<f32>>> {
        let frame_length = self.fs * self.frame_length_ms / 1000;
        let frame_shift = self.fs * self.frame_shift_ms / 1000;

        if waveform.len() < frame_length {
            return Ok(vec![]);
        }

        let mut frames = vec![];
        let num_frames = (waveform.len() - frame_length) / frame_shift + 1;

        let mut planner = FftPlanner::new();
        let fft_size = frame_length.next_power_of_two();
        let fft = planner.plan_fft_forward(fft_size);

        for i in 0..num_frames {
            let start = i * frame_shift;
            let mut frame: Vec<f32> = waveform[start..start + frame_length].to_vec();

            // Apply window
            for (j, w) in self.window.iter().enumerate() {
                frame[j] *= w;
            }

            // FFT
            let mut input: Vec<Complex<f32>> = frame
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .chain(std::iter::repeat(Complex::new(0.0, 0.0)).take(fft_size - frame_length))
                .collect();
            fft.process(&mut input);

            // Power spectrum
            let power_spectrum: Vec<f32> = input
                .iter()
                .take(fft_size / 2 + 1)
                .map(|c| c.norm_sqr())
                .collect();

            // Mel filtering
            let mut mel_feat = vec![0.0; self.n_mels];
            for (m, filter) in self.mel_filters.iter().enumerate() {
                let mut sum = 0.0;
                for (j, &w) in filter.iter().enumerate() {
                    sum += w * power_spectrum[j];
                }
                mel_feat[m] = (sum.max(1e-10)).ln();
            }
            frames.push(mel_feat);
        }

        Ok(frames)
    }

    pub fn apply_lfr(&self, fbank: &[Vec<f32>]) -> (Vec<f32>, usize) {
        let t = fbank.len();
        if t == 0 {
             return (vec![], 0);
        }
        let t_lfr = (t as f32 / self.lfr_n as f32).ceil() as usize;
        let left_padding_size = (self.lfr_m - 1) / 2;

        let mut padded_fbank = Vec::with_capacity(t + left_padding_size);
        for _ in 0..left_padding_size {
            padded_fbank.push(fbank[0].clone());
        }
        padded_fbank.extend_from_slice(fbank);

        let mut lfr_outputs = Vec::with_capacity(t_lfr * self.lfr_m * self.n_mels);
        for i in 0..t_lfr {
            let start_idx = i * self.lfr_n;
            for j in 0..self.lfr_m {
                let idx = start_idx + j;
                let frame = if idx < padded_fbank.len() {
                    &padded_fbank[idx]
                } else {
                    &padded_fbank[padded_fbank.len() - 1]
                };
                lfr_outputs.extend_from_slice(frame);
            }
        }
        (lfr_outputs, t_lfr)
    }

    #[allow(dead_code)]
    pub fn apply_cmvn(&self, feats: &mut [f32], means: &[f32], vars: &[f32]) {
        let dim = means.len();
        for chunk in feats.chunks_exact_mut(dim) {
            for i in 0..dim {
                chunk[i] = (chunk[i] + means[i]) * vars[i];
            }
        }
    }
}

fn hamming_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()
        })
        .collect()
}

fn create_mel_filters(n_mels: usize, frame_length: usize, fs: f32) -> Vec<Vec<f32>> {
    let fft_size = frame_length.next_power_of_two();
    let num_bins = fft_size / 2 + 1;
    let min_freq = 0.0;
    let max_freq = fs / 2.0;

    let min_mel = freq_to_mel(min_freq);
    let max_mel = freq_to_mel(max_freq);

    let mut mel_points = vec![0.0; n_mels + 2];
    for i in 0..n_mels + 2 {
        mel_points[i] = mel_to_freq(min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32);
    }

    let mut bins = vec![0; n_mels + 2];
    for i in 0..n_mels + 2 {
        bins[i] = (mel_points[i] * (fft_size + 1) as f32 / fs).floor() as usize;
    }

    let mut filters = vec![vec![0.0; num_bins]; n_mels];
    for i in 0..n_mels {
        for j in bins[i]..bins[i + 1] {
            filters[i][j] = (j - bins[i]) as f32 / (bins[i + 1] - bins[i]) as f32;
        }
        for j in bins[i + 1]..bins[i + 2] {
            filters[i][j] = (bins[i + 2] - j) as f32 / (bins[i + 2] - bins[i + 1]) as f32;
        }
    }
    filters
}

fn freq_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

fn mel_to_freq(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}
