//! Formula recognition with PP-FormulaNet_plus (image to LaTeX).
//!
//! The model is exported as an encoder (image to cross-attention keys/values) and a
//! KV-cached single-step decoder; greedy decoding runs here.

use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::{Array4, ArrayD, ArrayViewD, IxDyn};

use super::image_ops::{crop_rgb, resize_rgb, to_rgb8};
use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, MultiModel};

const INPUT_SIZE: u32 = 768;
const DECODER_START: u32 = 2;
const EOS: u32 = 2;
const LAST_SPECIAL_TOKEN: u32 = 22;
const PIXEL_MEAN: f32 = 0.7931;
const PIXEL_STD: f32 = 0.1738;

/// Byte-level BPE vocabulary used to turn token ids into text
#[derive(Debug, Clone)]
pub struct LatexVocab {
    tokens: Vec<String>,
}

impl LatexVocab {
    /// Load a vocabulary file with one token per line (line number = token id)
    pub fn from_file(path: impl AsRef<Path>) -> OcrResult<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Ok(Self::from_text(&text))
    }

    /// Parse vocabulary text with one token per line
    pub fn from_text(text: &str) -> Self {
        Self {
            tokens: text
                .lines()
                .map(|l| l.trim_end_matches('\r').to_string())
                .collect(),
        }
    }

    /// Vocabulary size
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Decode ids into text, skipping special tokens
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id <= LAST_SPECIAL_TOKEN {
                continue;
            }
            if let Some(token) = self.tokens.get(id as usize) {
                for ch in token.chars() {
                    if let Some(b) = unicode_to_byte(ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Inverse of GPT-2's `bytes_to_unicode` mapping
fn unicode_to_byte(ch: char) -> Option<u8> {
    let c = ch as u32;
    let direct =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    if c < 256 && direct(c) {
        return Some(c as u8);
    }
    if c >= 256 {
        // Bytes outside the printable set are mapped to 256.. in increasing byte order.
        let n = c - 256;
        let mut count = 0u32;
        for b in 0u32..256 {
            if !direct(b) {
                if count == n {
                    return Some(b as u8);
                }
                count += 1;
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// LaTeX post-processing (PaddleX / transformers `PPFormulaNetProcessor.post_process`)
// ---------------------------------------------------------------------------

fn is_cjk_ideograph(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// Unwrap `\text{...}` groups that contain Chinese characters, then drop double quotes
pub fn remove_chinese_text_wrapping(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && chars[i + 1..].starts_with(&['t', 'e', 'x', 't']) {
            let mut j = i + 5;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == '{' {
                if let Some(end) = chars[j + 1..].iter().position(|&c| c == '{' || c == '}') {
                    let close = j + 1 + end;
                    let inner = &chars[j + 1..close];
                    if chars[close] == '}' && inner.iter().any(|&c| is_cjk_ideograph(c)) {
                        out.extend(inner.iter());
                        i = close + 1;
                        continue;
                    }
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out.replace('"', "")
}

const TEXT_MACROS: [&str; 4] = ["operatorname", "mathrm", "text", "mathbf"];

/// Spans of `\(operatorname|mathrm|text|mathbf)\s?\*? {.*?}` in char indices
fn text_macro_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0;
    'outer: while i < chars.len() {
        if chars[i] == '\\' {
            for name in TEXT_MACROS {
                let name: Vec<char> = name.chars().collect();
                if chars[i + 1..].starts_with(&name) {
                    let j = i + 1 + name.len();
                    // `\s?\*? ` then `{`: try with and without the optional whitespace.
                    for skip_ws in [true, false] {
                        let mut k = j;
                        if skip_ws {
                            if k < chars.len() && chars[k].is_whitespace() {
                                k += 1;
                            } else {
                                continue;
                            }
                        }
                        if k < chars.len() && chars[k] == '*' {
                            k += 1;
                        }
                        if k + 1 < chars.len() && chars[k] == ' ' && chars[k + 1] == '{' {
                            if let Some(end) =
                                chars[k + 2..].iter().position(|&c| c == '}' || c == '\n')
                            {
                                if chars[k + 2 + end] == '}' {
                                    let stop = k + 2 + end + 1;
                                    spans.push((i, stop));
                                    i = stop;
                                    continue 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
        i += 1;
    }
    spans
}

/// Macros followed by a space and a word character inside a span (`(\\[a-zA-Z]+)\s(?=\w)`)
fn spaced_macros(chars: &[char]) -> Vec<String> {
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_alphabetic() {
                j += 1;
            }
            if j > i + 1 && j < chars.len() && chars[j].is_whitespace() {
                let next = chars.get(j + 1).copied();
                if next
                    .map(|c| c.is_alphanumeric() || c == '_')
                    .unwrap_or(false)
                {
                    found.push(chars[i..j].iter().collect());
                    i = j + 1;
                    continue;
                }
                if next == Some('}') {
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    found
}

fn is_noletter(c: char) -> bool {
    !c.is_alphanumeric() || c.is_numeric()
}

fn is_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
}

/// One pass of `(?!\\ )(left)\s+?(right)` -> `\1\2`
fn remove_space_between(
    text: &str,
    left: fn(char) -> bool,
    right: fn(char) -> bool,
    guard: bool,
) -> String {
    let s: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        let guarded = guard && c == '\\' && s.get(i + 1) == Some(&' ');
        if left(c) && !guarded {
            let mut k = 1;
            let mut matched = None;
            while i + k < s.len() && s[i + k].is_whitespace() {
                if let Some(&y) = s.get(i + k + 1) {
                    if right(y) {
                        matched = Some((k, y));
                        break;
                    }
                }
                k += 1;
            }
            if let Some((k, y)) = matched {
                out.push(c);
                out.push(y);
                i += k + 2;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Normalize spacing of generated LaTeX like PaddleX
pub fn normalize_latex(text: &str) -> String {
    let mut text = text.to_string();
    let original: Vec<char> = text.chars().collect();
    // PaddleX quirk: protecting one macro inside a text-style group removes every other space.
    for (start, end) in text_macro_spans(&original) {
        for m in spaced_macros(&original[start..end]) {
            let protected = TEXT_MACROS.iter().any(|p| m == format!("\\{}", p));
            if !protected && !m.trim().is_empty() {
                text = text.replace(&m, &format!("{}XXXXXXX", m));
                text = text.replace(' ', "");
            }
        }
    }
    loop {
        let prev = text.clone();
        text = remove_space_between(&text, is_noletter, is_noletter, true);
        text = remove_space_between(&text, is_noletter, is_letter, true);
        text = remove_space_between(&text, is_letter, is_noletter, false);
        if text == prev {
            break;
        }
    }
    text.replace("XXXXXXX", " ")
}

/// Full LaTeX post-processing applied to decoded text
pub fn postprocess_latex(text: &str) -> String {
    normalize_latex(&remove_chinese_text_wrapping(text))
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

fn luma(p: &Rgb<u8>) -> u8 {
    let [r, g, b] = p.0;
    ((r as u32 * 299 + g as u32 * 587 + b as u32 * 114 + 500) / 1000) as u8
}

/// Crop away the near-white margin around the formula
pub(crate) fn crop_margin(image: &RgbImage) -> RgbImage {
    let (min, max) = image.pixels().fold((255u8, 0u8), |(lo, hi), p| {
        let v = luma(p);
        (lo.min(v), hi.max(v))
    });
    if min == max {
        return image.clone();
    }
    let range = (max - min) as f32;
    let (mut x1, mut y1, mut x2, mut y2) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for (x, y, p) in image.enumerate_pixels() {
        if (luma(p) - min) as f32 / range * 255.0 < 200.0 {
            x1 = x1.min(x);
            y1 = y1.min(y);
            x2 = x2.max(x);
            y2 = y2.max(y);
        }
    }
    if x1 > x2 || y1 > y2 {
        return image.clone();
    }
    crop_rgb(image, x1, y1, x2 - x1 + 1, y2 - y1 + 1)
}

/// Target size after shortest-edge resize to 768 followed by thumbnailing into 768x768
pub(crate) fn target_size(width: u32, height: u32) -> (u32, u32) {
    let size = INPUT_SIZE as f64;
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let (rw, rh) = if w <= h {
        (size, (size * h / w).trunc())
    } else {
        ((size * w / h).trunc(), size)
    };
    let (mut tw, mut th) = (rw.min(size), rh.min(size));
    if rh > rw {
        tw = (rw * th / rh).trunc();
    } else if rw > rh {
        th = (rh * tw / rw).trunc();
    }
    ((tw as u32).max(1), (th as u32).max(1))
}

/// Build the `[1, 3, 768, 768]` encoder input
pub(crate) fn preprocess_formula(image: &DynamicImage) -> OcrResult<Array4<f32>> {
    let rgb = to_rgb8(image);
    if rgb.width() == 0 || rgb.height() == 0 {
        return Err(OcrError::InvalidParameter(
            "Formula image is empty".to_string(),
        ));
    }
    let cropped = crop_margin(&rgb);
    let (tw, th) = target_size(cropped.width(), cropped.height());
    let resized = resize_rgb(&cropped, tw, th, FilterType::CatmullRom);
    let left = (INPUT_SIZE - tw) / 2;
    let top = (INPUT_SIZE - th) / 2;
    let size = INPUT_SIZE as usize;
    // Black padding, normalized like the image content.
    let pad = (0.0 - PIXEL_MEAN) / PIXEL_STD;
    let mut tensor = Array4::<f32>::from_elem((1, 3, size, size), pad);
    for (x, y, p) in resized.enumerate_pixels() {
        let [r, g, b] = p.0;
        // PaddleX converts with OpenCV BGR2GRAY on RGB data, so the weights are swapped.
        let gray = 0.114 * r as f32 + 0.587 * g as f32 + 0.299 * b as f32;
        let v = (gray / 255.0 - PIXEL_MEAN) / PIXEL_STD;
        let (yy, xx) = ((y + top) as usize, (x + left) as usize);
        for c in 0..3 {
            tensor[[0, c, yy, xx]] = v;
        }
    }
    Ok(tensor)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Formula recognition result
#[derive(Debug, Clone, PartialEq)]
pub struct FormulaResult {
    /// Normalized LaTeX
    pub latex: String,
    /// Number of generated tokens (excluding the start token)
    pub token_count: usize,
}

/// PP-FormulaNet_plus formula recognizer
pub struct FormulaModel {
    encoder: MultiModel,
    decoder: MultiModel,
    vocab: LatexVocab,
    max_new_tokens: usize,
}

impl FormulaModel {
    /// Load encoder, decoder step and vocabulary files
    pub fn from_files(
        encoder_path: impl AsRef<Path>,
        decoder_path: impl AsRef<Path>,
        vocab_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let encoder = std::fs::read(encoder_path.as_ref())?;
        let decoder = std::fs::read(decoder_path.as_ref())?;
        let vocab = LatexVocab::from_file(vocab_path)?;
        Self::from_bytes(&encoder, &decoder, vocab, config)
    }

    /// Load from model bytes and a vocabulary
    pub fn from_bytes(
        encoder: &[u8],
        decoder: &[u8],
        vocab: LatexVocab,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        if vocab.is_empty() {
            return Err(OcrError::ModelLoadError(
                "Formula vocabulary is empty".to_string(),
            ));
        }
        let encoder = MultiModel::from_buffer(
            encoder,
            &["pixel_values"],
            &["cross_k", "cross_v"],
            config.clone(),
        )?;
        let decoder = MultiModel::from_buffer(
            decoder,
            &[
                "token",
                "position",
                "past_k",
                "past_v",
                "past_mask",
                "cross_k",
                "cross_v",
            ],
            &["next_token", "new_k", "new_v"],
            config,
        )?;
        Ok(Self {
            encoder,
            decoder,
            vocab,
            max_new_tokens: 1536,
        })
    }

    /// Limit the number of generated tokens (default 1536)
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens.max(1);
        self
    }

    /// Recognize a formula image
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<FormulaResult> {
        let pixels = preprocess_formula(image)?;
        let enc = self.encoder.run(&[pixels.view().into_dyn()])?;
        if enc.len() < 2 || enc[0].shape.len() != 4 || enc[0].shape != enc[1].shape {
            return Err(OcrError::PostprocessError(
                "Unexpected formula encoder output".to_string(),
            ));
        }
        let kv_shape = enc[0].shape.clone(); // [layers, heads, 144, head_dim]
        let row: usize = kv_shape[0] * kv_shape[1] * kv_shape[3];
        let cross_k = enc[0].clone().into_f32_array()?;
        let cross_v = enc[1].clone().into_f32_array()?;

        // Slot 0 of the cache is a masked dummy row so tensors never have a zero-length axis.
        let mut past_k = vec![0f32; row];
        let mut past_v = vec![0f32; row];
        let mut mask = vec![-1e9f32];
        let mut ids: Vec<u32> = vec![DECODER_START];

        for step in 0..self.max_new_tokens {
            let p = mask.len();
            let cache_shape = [p, kv_shape[0], kv_shape[1], kv_shape[3]];
            let token =
                ArrayD::from_elem(IxDyn(&[1]), *ids.last().unwrap_or(&DECODER_START) as f32);
            let position = ArrayD::from_elem(IxDyn(&[1]), step as f32);
            let pk = ArrayViewD::from_shape(IxDyn(&cache_shape), &past_k)
                .map_err(|e| OcrError::PostprocessError(e.to_string()))?;
            let pv = ArrayViewD::from_shape(IxDyn(&cache_shape), &past_v)
                .map_err(|e| OcrError::PostprocessError(e.to_string()))?;
            let pm = ArrayViewD::from_shape(IxDyn(&[p]), &mask)
                .map_err(|e| OcrError::PostprocessError(e.to_string()))?;
            let out = self.decoder.run(&[
                token.view(),
                position.view(),
                pk,
                pv,
                pm,
                cross_k.view(),
                cross_v.view(),
            ])?;
            if out.len() < 3 || out[1].len() != row || out[2].len() != row {
                return Err(OcrError::PostprocessError(
                    "Unexpected formula decoder output".to_string(),
                ));
            }
            let mut next = out[0].to_f32_vec().first().copied().unwrap_or(EOS as f32) as u32;
            if step + 1 == self.max_new_tokens {
                next = EOS;
            }
            ids.push(next);
            if next == EOS {
                break;
            }
            past_k.extend(out[1].to_f32_vec());
            past_v.extend(out[2].to_f32_vec());
            mask.push(0.0);
        }

        let generated = &ids[1..];
        let raw = self.vocab.decode(generated);
        Ok(FormulaResult {
            latex: postprocess_latex(&raw),
            token_count: generated.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicode_to_byte_roundtrip() {
        assert_eq!(unicode_to_byte('a'), Some(b'a'));
        assert_eq!(unicode_to_byte('Ġ'), Some(b' '));
        assert_eq!(unicode_to_byte('Ċ'), Some(b'\n'));
        assert_eq!(unicode_to_byte('\u{0100}'), Some(0));
        assert_eq!(unicode_to_byte('é'), Some(0xE9));
    }

    #[test]
    fn test_vocab_decode_skips_special_and_joins_bytes() {
        let mut lines: Vec<String> = (0..23).map(|i| format!("<special{}>", i)).collect();
        lines.extend(["\\frac".to_string(), "Ġ{".to_string(), "Ã©".to_string()]);
        let vocab = LatexVocab::from_text(&lines.join("\n"));
        assert_eq!(vocab.len(), 26);
        assert_eq!(vocab.decode(&[2, 23, 24, 2]), "\\frac {");
        // "Ã©" is the byte-level encoding of the UTF-8 bytes of "é".
        assert_eq!(vocab.decode(&[25]), "é");
    }

    #[test]
    fn test_normalize_latex_removes_spaces_like_paddlex() {
        let raw = "\\zeta _ { 0 } ( \\nu ) = - \\frac { \\nu \\varrho ^ { - 2 \\nu } } { \\pi }";
        assert_eq!(
            normalize_latex(raw),
            "\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}"
        );
        // Spaces between two letters are kept.
        assert_eq!(normalize_latex("d z \\quad ,"), "d z\\quad,");
        assert_eq!(normalize_latex("a  +  b"), "a+b");
    }

    #[test]
    fn test_normalize_latex_keeps_escaped_space() {
        assert_eq!(normalize_latex("x \\ y"), "x\\ y");
    }

    #[test]
    fn test_normalize_latex_text_macro_quirk() {
        // A non-protected macro followed by a word inside \mathrm{...} keeps one space.
        assert_eq!(
            normalize_latex("\\mathrm { \\Delta x }"),
            "\\mathrm{\\Delta x}"
        );
    }

    #[test]
    fn test_remove_chinese_text_wrapping() {
        assert_eq!(
            remove_chinese_text_wrapping("a \\text { 面积 } = \"b\""),
            "a  面积  = b"
        );
        assert_eq!(remove_chinese_text_wrapping("\\text{abc}"), "\\text{abc}");
    }

    #[test]
    fn test_target_size() {
        assert_eq!(target_size(448, 64), (768, 109));
        assert_eq!(target_size(64, 448), (109, 768));
        assert_eq!(target_size(100, 100), (768, 768));
    }

    #[test]
    fn test_crop_margin() {
        let mut img = RgbImage::from_pixel(10, 10, Rgb([255, 255, 255]));
        img.put_pixel(3, 4, Rgb([0, 0, 0]));
        img.put_pixel(6, 5, Rgb([0, 0, 0]));
        assert_eq!(crop_margin(&img).dimensions(), (4, 2));
        let blank = RgbImage::from_pixel(5, 5, Rgb([255, 255, 255]));
        assert_eq!(crop_margin(&blank).dimensions(), (5, 5));
    }

    #[test]
    fn test_preprocess_formula_shape_and_padding() {
        let mut img = RgbImage::from_pixel(40, 10, Rgb([255, 255, 255]));
        for x in 0..40 {
            img.put_pixel(x, 5, Rgb([0, 0, 0]));
        }
        img.put_pixel(0, 0, Rgb([0, 0, 0]));
        let t = preprocess_formula(&DynamicImage::ImageRgb8(img)).unwrap();
        assert_eq!(t.shape(), &[1, 3, 768, 768]);
        let pad = (0.0 - PIXEL_MEAN) / PIXEL_STD;
        assert!((t[[0, 0, 0, 0]] - pad).abs() < 1e-5);
        assert_eq!(t[[0, 0, 384, 384]], t[[0, 2, 384, 384]]);
    }
}
