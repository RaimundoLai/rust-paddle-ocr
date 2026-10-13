//! Table recognition: SLANet_plus structure, table type classification, cell detection,
//! and filling recognized text into HTML.
//!
//! SLANet's autoregressive decoder is exported as two graphs (`encoder` and one decode
//! `step`) so the loop runs here instead of inside the model, which MNN cannot execute.

use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use ndarray::{Array1, Array2, ArrayD, IxDyn};

use super::detr::DetrModel;
use super::geometry::BoundingBox;
use super::image_ops::{
    crop_rgb, resize_rgb, round_half_even, to_nchw, to_nchw_padded, to_rgb8, ChannelOrder,
    Normalization,
};
use super::text_lines::{overlaps_region, TextLine};
use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, MultiModel, Tensor};

/// SLANet structure vocabulary (merged `<td></td>` variant)
pub const TABLE_STRUCTURE_TOKENS: [&str; 50] = [
    "sos",
    "<thead>",
    "</thead>",
    "<tbody>",
    "</tbody>",
    "<tr>",
    "</tr>",
    "<td",
    ">",
    "</td>",
    " colspan=\"2\"",
    " colspan=\"3\"",
    " colspan=\"4\"",
    " colspan=\"5\"",
    " colspan=\"6\"",
    " colspan=\"7\"",
    " colspan=\"8\"",
    " colspan=\"9\"",
    " colspan=\"10\"",
    " colspan=\"11\"",
    " colspan=\"12\"",
    " colspan=\"13\"",
    " colspan=\"14\"",
    " colspan=\"15\"",
    " colspan=\"16\"",
    " colspan=\"17\"",
    " colspan=\"18\"",
    " colspan=\"19\"",
    " colspan=\"20\"",
    " rowspan=\"2\"",
    " rowspan=\"3\"",
    " rowspan=\"4\"",
    " rowspan=\"5\"",
    " rowspan=\"6\"",
    " rowspan=\"7\"",
    " rowspan=\"8\"",
    " rowspan=\"9\"",
    " rowspan=\"10\"",
    " rowspan=\"11\"",
    " rowspan=\"12\"",
    " rowspan=\"13\"",
    " rowspan=\"14\"",
    " rowspan=\"15\"",
    " rowspan=\"16\"",
    " rowspan=\"17\"",
    " rowspan=\"18\"",
    " rowspan=\"19\"",
    " rowspan=\"20\"",
    "<td></td>",
    "eos",
];

const SOS: usize = 0;
const EOS: usize = 49;
const TD_OPEN: usize = 7;
const TD_EMPTY: usize = 48;
const TABLE_INPUT_SIZE: u32 = 488;
const HIDDEN_SIZE: usize = 256;

/// Predicted table structure for one table crop
#[derive(Debug, Clone, PartialEq)]
pub struct TableStructure {
    /// Structure tokens between `<table>` and `</table>`
    pub tokens: Vec<String>,
    /// One cell polygon per `<td` / `<td></td>` token, `x1,y1,...,x4,y4` in crop pixels
    pub cells: Vec<[f32; 8]>,
    /// Mean token probability
    pub score: f32,
}

impl TableStructure {
    /// Cell polygons as axis-aligned boxes in crop coordinates
    pub fn cell_boxes(&self) -> Vec<BoundingBox> {
        self.cells
            .iter()
            .map(|c| {
                let pts = [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]];
                BoundingBox::from_points(&pts).unwrap_or_default()
            })
            .collect()
    }

    /// Number of cells (closing `</td>` tokens)
    pub fn cell_count(&self) -> usize {
        self.tokens.iter().filter(|t| t.contains("</td>")).count()
    }

    /// HTML skeleton without text
    pub fn to_html(&self) -> String {
        format!(
            "<html><body><table>{}</table></body></html>",
            self.tokens.concat()
        )
    }
}

/// Decode per-step SLANet outputs into tokens, cells and score.
///
/// `steps` holds `(probabilities[50], loc[8])` for each decoded step; `scale` maps the
/// normalized locations back to crop pixels.
pub(crate) fn decode_table_steps(
    steps: &[(Vec<f32>, [f32; 8])],
    scale: (f64, f64),
) -> TableStructure {
    let mut tokens = Vec::new();
    let mut cells = Vec::new();
    let mut scores = Vec::new();
    for (t, (probs, loc)) in steps.iter().enumerate() {
        let (idx, prob) = argmax(probs);
        if t > 0 && idx == EOS {
            break;
        }
        if idx == SOS || idx == EOS {
            continue;
        }
        if idx == TD_OPEN || idx == TD_EMPTY {
            let mut cell = [0f32; 8];
            for k in 0..8 {
                let s = if k % 2 == 0 { scale.0 } else { scale.1 };
                cell[k] = (loc[k] as f64 * s).trunc() as f32;
            }
            cells.push(cell);
        }
        tokens.push(TABLE_STRUCTURE_TOKENS[idx].to_string());
        scores.push(prob);
    }
    let score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f32>() / scores.len() as f32
    };
    TableStructure {
        tokens,
        cells,
        score,
    }
}

fn argmax(values: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in values.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|v| v / sum).collect()
}

fn first_f32(outputs: &[Tensor], index: usize, name: &str) -> OcrResult<Vec<f32>> {
    outputs
        .get(index)
        .map(|t| t.to_f32_vec())
        .ok_or_else(|| OcrError::PostprocessError(format!("Missing table model output {}", name)))
}

/// SLANet_plus table structure recognizer
pub struct TableStructureModel {
    encoder: MultiModel,
    step: MultiModel,
    max_steps: usize,
}

impl TableStructureModel {
    /// Load from the encoder and step model files
    pub fn from_files(
        encoder_path: impl AsRef<Path>,
        step_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let encoder = std::fs::read(encoder_path.as_ref())?;
        let step = std::fs::read(step_path.as_ref())?;
        Self::from_bytes(&encoder, &step, config)
    }

    /// Load from model bytes
    pub fn from_bytes(
        encoder: &[u8],
        step: &[u8],
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let encoder = MultiModel::from_buffer(
            encoder,
            &["x"],
            &["p2o.pd_op.transpose.0.0"],
            config.clone(),
        )?;
        let step = MultiModel::from_buffer(
            step,
            &["features", "hidden", "prev_char"],
            &["hidden_out", "structure_logits", "loc"],
            config,
        )?;
        Ok(Self {
            encoder,
            step,
            max_steps: 500,
        })
    }

    /// Limit the number of decoded structure tokens (default 500)
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    /// Recognize the structure of a table crop
    pub fn predict(&self, table_image: &DynamicImage) -> OcrResult<TableStructure> {
        let rgb = to_rgb8(table_image);
        let (w, h) = rgb.dimensions();
        if w == 0 || h == 0 {
            return Err(OcrError::InvalidParameter(
                "Table image is empty".to_string(),
            ));
        }
        let size = TABLE_INPUT_SIZE as f64;
        let ratio = size / w.max(h) as f64;
        let nw = (round_half_even(w as f64 * ratio) as u32).max(1);
        let nh = (round_half_even(h as f64 * ratio) as u32).max(1);
        let resized = resize_rgb(&rgb, nw, nh, FilterType::Triangle);
        let input = to_nchw_padded(
            &resized,
            ChannelOrder::Bgr,
            &Normalization::IMAGENET,
            TABLE_INPUT_SIZE,
            TABLE_INPUT_SIZE,
            0.0,
        );

        let enc_out = self.encoder.run(&[input.view().into_dyn()])?;
        let feat = enc_out.into_iter().next().ok_or_else(|| {
            OcrError::PostprocessError("Table encoder returned no output".to_string())
        })?;
        let feat_shape = feat.shape.clone();
        let features: ArrayD<f32> = feat.into_f32_array()?;
        if feat_shape.len() != 3 {
            return Err(OcrError::PostprocessError(format!(
                "Unexpected table feature shape {:?}",
                feat_shape
            )));
        }

        let mut hidden = Array2::<f32>::zeros((1, HIDDEN_SIZE)).into_dyn();
        let mut prev = Array1::<f32>::zeros(1).into_dyn();
        let mut steps: Vec<(Vec<f32>, [f32; 8])> = Vec::new();
        for t in 0..=self.max_steps {
            let out = self
                .step
                .run(&[features.view(), hidden.view(), prev.view()])?;
            let new_hidden = first_f32(&out, 0, "hidden_out")?;
            let logits = first_f32(&out, 1, "structure_logits")?;
            let loc_v = first_f32(&out, 2, "loc")?;
            if new_hidden.len() != HIDDEN_SIZE
                || logits.len() != TABLE_STRUCTURE_TOKENS.len()
                || loc_v.len() != 8
            {
                return Err(OcrError::PostprocessError(
                    "Unexpected table step output size".to_string(),
                ));
            }
            let probs = softmax(&logits);
            let (idx, _) = argmax(&probs);
            let mut loc = [0f32; 8];
            loc.copy_from_slice(&loc_v);
            steps.push((probs, loc));
            hidden = ArrayD::from_shape_vec(IxDyn(&[1, HIDDEN_SIZE]), new_hidden)
                .map_err(|e| OcrError::PostprocessError(e.to_string()))?;
            prev = ArrayD::from_elem(IxDyn(&[1]), idx as f32);
            if t > 0 && idx == EOS {
                break;
            }
        }

        // Locations are normalized to the padded canvas; the long side maps to 488 px.
        let r = (size / w as f64).min(size / h as f64);
        let scale = (size / r, size / r);
        Ok(decode_table_steps(&steps, scale))
    }
}

/// Table style predicted by the classifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableType {
    /// Tables with ruling lines
    Wired,
    /// Tables without ruling lines
    Wireless,
}

/// PP-LCNet_x1_0_table_cls classifier
pub struct TableClassifier {
    model: MultiModel,
}

impl TableClassifier {
    /// Load from a file
    pub fn from_file(path: impl AsRef<Path>, config: Option<InferenceConfig>) -> OcrResult<Self> {
        Self::from_bytes(&std::fs::read(path.as_ref())?, config)
    }

    /// Load from bytes
    pub fn from_bytes(bytes: &[u8], config: Option<InferenceConfig>) -> OcrResult<Self> {
        Ok(Self {
            model: MultiModel::from_buffer(bytes, &["x"], &["fetch_name_0"], config)?,
        })
    }

    /// Classify a table crop, returning the type and its score
    pub fn classify(&self, table_image: &DynamicImage) -> OcrResult<(TableType, f32)> {
        let rgb = to_rgb8(table_image);
        let (w, h) = rgb.dimensions();
        if w == 0 || h == 0 {
            return Err(OcrError::InvalidParameter(
                "Table image is empty".to_string(),
            ));
        }
        let scale = 256.0 / w.min(h) as f64;
        let nw = (round_half_even(w as f64 * scale) as u32).max(224);
        let nh = (round_half_even(h as f64 * scale) as u32).max(224);
        let resized = resize_rgb(&rgb, nw, nh, FilterType::Triangle);
        let cropped = crop_rgb(&resized, (nw - 224) / 2, (nh - 224) / 2, 224, 224);
        let input = to_nchw(&cropped, ChannelOrder::Rgb, &Normalization::IMAGENET);
        let out = self.model.run(&[input.view().into_dyn()])?;
        let scores = first_f32(&out, 0, "fetch_name_0")?;
        if scores.len() < 2 {
            return Err(OcrError::PostprocessError(
                "Unexpected table classifier output".to_string(),
            ));
        }
        Ok(if scores[0] >= scores[1] {
            (TableType::Wired, scores[0])
        } else {
            (TableType::Wireless, scores[1])
        })
    }
}

/// RT-DETR table cell detector (wired or wireless variant)
pub struct TableCellDetector {
    detector: DetrModel,
    score_threshold: f32,
    nms_threshold: f32,
}

impl TableCellDetector {
    /// Load from a file
    pub fn from_file(path: impl AsRef<Path>, config: Option<InferenceConfig>) -> OcrResult<Self> {
        Self::from_bytes(&std::fs::read(path.as_ref())?, config)
    }

    /// Load from bytes
    pub fn from_bytes(bytes: &[u8], config: Option<InferenceConfig>) -> OcrResult<Self> {
        Ok(Self {
            detector: DetrModel::from_bytes(bytes, 2, 640, config)?,
            score_threshold: 0.3,
            nms_threshold: 0.3,
        })
    }

    /// Detect cells in a table crop, returning boxes in crop coordinates (score-descending)
    pub fn detect(&self, table_image: &DynamicImage) -> OcrResult<Vec<(BoundingBox, f32)>> {
        let rgb = to_rgb8(table_image);
        let (w, h) = (rgb.width() as f32, rgb.height() as f32);
        let output = self.detector.detect(&rgb)?;
        let cells: Vec<(BoundingBox, f32)> = output
            .rows
            .iter()
            .filter(|r| r.score > self.score_threshold)
            .map(|r| (r.bbox.clip(w, h), r.score))
            .filter(|(b, _)| b.is_valid())
            .collect();
        Ok(cell_nms(cells, self.nms_threshold))
    }
}

/// Greedy NMS over cell boxes (plain IoU)
pub(crate) fn cell_nms(
    mut cells: Vec<(BoundingBox, f32)>,
    iou_threshold: f32,
) -> Vec<(BoundingBox, f32)> {
    cells.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<(BoundingBox, f32)> = Vec::new();
    for c in cells {
        if keep.iter().all(|k| k.0.iou(&c.0) <= iou_threshold) {
            keep.push(c);
        }
    }
    keep
}

/// Sort detected cells row-major: new row when the top edge moves more than 10 px
pub(crate) fn sort_cells_row_major(cells: &[BoundingBox]) -> Vec<BoundingBox> {
    let mut sorted = cells.to_vec();
    sorted.sort_by(|a, b| a.y1.partial_cmp(&b.y1).unwrap_or(std::cmp::Ordering::Equal));
    let mut rows: Vec<Vec<BoundingBox>> = Vec::new();
    for c in sorted {
        match rows.last_mut() {
            Some(row) if (c.y1 - row[0].y1).abs() <= 10.0 => row.push(c),
            _ => rows.push(vec![c]),
        }
    }
    rows.into_iter()
        .flat_map(|mut row| {
            row.sort_by(|a, b| a.x1.partial_cmp(&b.x1).unwrap_or(std::cmp::Ordering::Equal));
            row
        })
        .collect()
}

/// PaddleX table matching distance between a cell and an OCR box
fn cell_distance(cell: &BoundingBox, ocr: &BoundingBox) -> f32 {
    let tl = (ocr.x1 - cell.x1).abs() + (ocr.y1 - cell.y1).abs();
    let br = (ocr.x2 - cell.x2).abs() + (ocr.y2 - cell.y2).abs();
    tl + br + tl.min(br)
}

/// Assign every OCR box to its best cell: highest IoU, then smallest distance
pub(crate) fn match_ocr_to_cells(cells: &[BoundingBox], ocr: &[BoundingBox]) -> Vec<Vec<usize>> {
    let mut matched = vec![Vec::new(); cells.len()];
    if cells.is_empty() {
        return matched;
    }
    for (i, o) in ocr.iter().enumerate() {
        let mut best = 0usize;
        let mut best_key = (f32::INFINITY, f32::INFINITY);
        for (j, c) in cells.iter().enumerate() {
            let key = (1.0 - c.iou(o), cell_distance(c, o));
            if key.0 < best_key.0 || (key.0 == best_key.0 && key.1 < best_key.1) {
                best_key = key;
                best = j;
            }
        }
        matched[best].push(i);
    }
    matched
}

/// Fill matched texts into the structure tokens (PaddleX `get_html_result`)
pub(crate) fn fill_html(tokens: &[String], matched: &[Vec<usize>], texts: &[String]) -> String {
    let mut html = String::from("<html><body><table>");
    let mut td_index = 0usize;
    for tag in tokens {
        if !tag.contains("</td>") {
            html.push_str(tag);
            continue;
        }
        if tag == "<td></td>" {
            html.push_str("<td>");
        }
        if let Some(idxs) = matched.get(td_index).filter(|v| !v.is_empty()) {
            let multi = idxs.len() > 1;
            let bold = multi && texts[idxs[0]].contains("<b>");
            if bold {
                html.push_str("<b>");
            }
            for (k, &oi) in idxs.iter().enumerate() {
                let mut content: &str = &texts[oi];
                let owned;
                if multi {
                    if content.is_empty() {
                        continue;
                    }
                    if let Some(stripped) = content.strip_prefix(' ') {
                        content = stripped;
                    }
                    if content.contains("<b>") {
                        content = content.get(3..).unwrap_or("");
                    }
                    if content.contains("</b>") {
                        content = content.get(..content.len().saturating_sub(4)).unwrap_or("");
                    }
                    if content.is_empty() {
                        continue;
                    }
                    if k != idxs.len() - 1 && !content.ends_with(' ') {
                        owned = format!("{} ", content);
                        content = &owned;
                    }
                }
                html.push_str(content);
            }
            if bold {
                html.push_str("</b>");
            }
        }
        html.push_str(if tag == "<td></td>" { "</td>" } else { tag });
        td_index += 1;
    }
    html.push_str("</table></body></html>");
    html
}

/// Result of recognizing one table
#[derive(Debug, Clone, PartialEq)]
pub struct TableResult {
    /// HTML with recognized text
    pub html: String,
    /// Cell boxes in page coordinates
    pub cells: Vec<BoundingBox>,
    /// Structure confidence
    pub score: f32,
    /// Table type when a classifier was used
    pub table_type: Option<TableType>,
}

/// Table recognizer combining structure, optional classification and cell detection
pub struct TableRecognizer {
    structure: TableStructureModel,
    classifier: Option<TableClassifier>,
    wired_cells: Option<TableCellDetector>,
    wireless_cells: Option<TableCellDetector>,
}

impl TableRecognizer {
    /// Create a recognizer from a structure model
    pub fn new(structure: TableStructureModel) -> Self {
        Self {
            structure,
            classifier: None,
            wired_cells: None,
            wireless_cells: None,
        }
    }

    /// Add a wired/wireless table classifier
    pub fn with_classifier(mut self, classifier: TableClassifier) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Add cell detectors used to refine cell boxes for each table type
    pub fn with_cell_detectors(
        mut self,
        wired: Option<TableCellDetector>,
        wireless: Option<TableCellDetector>,
    ) -> Self {
        self.wired_cells = wired;
        self.wireless_cells = wireless;
        self
    }

    /// Structure model
    pub fn structure_model(&self) -> &TableStructureModel {
        &self.structure
    }

    /// Recognize the table inside `table_box` of `page`, filling text from page-level `lines`
    pub fn recognize(
        &self,
        page: &RgbImage,
        table_box: &BoundingBox,
        lines: &[TextLine],
    ) -> OcrResult<TableResult> {
        let (pw, ph) = (page.width() as f32, page.height() as f32);
        let x0 = table_box.x1.max(0.0).trunc();
        let y0 = table_box.y1.max(0.0).trunc();
        let (x, y, w, h) = table_box
            .to_pixel_rect(page.width(), page.height())
            .ok_or_else(|| {
                OcrError::InvalidParameter("Table box is outside the page".to_string())
            })?;
        let crop = DynamicImage::ImageRgb8(crop_rgb(page, x, y, w, h));

        let table_type = match &self.classifier {
            Some(c) => Some(c.classify(&crop)?.0),
            None => None,
        };
        let structure = self.structure.predict(&crop)?;
        let mut crop_cells = structure.cell_boxes();

        let detector = match table_type {
            Some(TableType::Wired) => self.wired_cells.as_ref(),
            Some(TableType::Wireless) => self.wireless_cells.as_ref(),
            None => self.wired_cells.as_ref().or(self.wireless_cells.as_ref()),
        };
        if let Some(det) = detector {
            let detected: Vec<BoundingBox> =
                det.detect(&crop)?.into_iter().map(|(b, _)| b).collect();
            // Detected cells are more precise but carry no span information; use them only
            // when they line up one-to-one with the predicted cells.
            if !detected.is_empty() && detected.len() == structure.cell_count() {
                crop_cells = sort_cells_row_major(&detected);
            }
        }

        let cells: Vec<BoundingBox> = crop_cells
            .iter()
            .map(|c| c.offset(x0, y0).clip(pw, ph))
            .collect();
        let table_lines: Vec<&TextLine> = lines
            .iter()
            .filter(|l| overlaps_region(&l.bbox, table_box))
            .collect();
        let html = if cells.is_empty() || table_lines.is_empty() {
            structure.to_html()
        } else {
            let ocr_boxes: Vec<BoundingBox> = table_lines.iter().map(|l| l.bbox).collect();
            let texts: Vec<String> = table_lines.iter().map(|l| l.text.clone()).collect();
            let matched = match_ocr_to_cells(&cells, &ocr_boxes);
            fill_html(&structure.tokens, &matched, &texts)
        };
        Ok(TableResult {
            html,
            cells,
            score: structure.score,
            table_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn onehot(idx: usize, p: f32) -> Vec<f32> {
        let mut v = vec![(1.0 - p) / 49.0; 50];
        v[idx] = p;
        v
    }

    fn toks(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_vocabulary_layout() {
        assert_eq!(TABLE_STRUCTURE_TOKENS.len(), 50);
        assert_eq!(TABLE_STRUCTURE_TOKENS[10], " colspan=\"2\"");
        assert_eq!(TABLE_STRUCTURE_TOKENS[28], " colspan=\"20\"");
        assert_eq!(TABLE_STRUCTURE_TOKENS[29], " rowspan=\"2\"");
        assert_eq!(TABLE_STRUCTURE_TOKENS[47], " rowspan=\"20\"");
        assert_eq!(TABLE_STRUCTURE_TOKENS[48], "<td></td>");
    }

    #[test]
    fn test_decode_table_steps() {
        let loc = [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4];
        let steps = vec![
            (onehot(3, 0.9), [0.0; 8]),  // <tbody>
            (onehot(5, 0.9), [0.0; 8]),  // <tr>
            (onehot(48, 0.9), loc),      // <td></td> with box
            (onehot(7, 0.8), loc),       // <td
            (onehot(10, 0.9), [0.0; 8]), //  colspan="2"
            (onehot(8, 0.9), [0.0; 8]),  // >
            (onehot(9, 0.9), [0.0; 8]),  // </td>
            (onehot(6, 0.9), [0.0; 8]),  // </tr>
            (onehot(49, 0.9), [0.0; 8]), // eos
            (onehot(5, 0.9), [0.0; 8]),  // ignored after eos
        ];
        let s = decode_table_steps(&steps, (100.0, 100.0));
        assert_eq!(
            s.tokens.concat(),
            "<tbody><tr><td></td><td colspan=\"2\"></td></tr>"
        );
        assert_eq!(s.cells.len(), 2);
        assert_eq!(s.cells[0], [10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 10.0, 40.0]);
        assert_eq!(s.cell_count(), 2);
        assert_eq!(s.cell_boxes()[0], BoundingBox::new(10.0, 20.0, 30.0, 40.0));
        assert!((s.score - (0.9 * 7.0 + 0.8) / 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_decode_ignores_leading_eos_and_sos() {
        let steps = vec![
            (onehot(49, 0.9), [0.0; 8]),
            (onehot(0, 0.9), [0.0; 8]),
            (onehot(5, 0.9), [0.0; 8]),
        ];
        let s = decode_table_steps(&steps, (1.0, 1.0));
        assert_eq!(s.tokens, toks(&["<tr>"]));
        assert!(decode_table_steps(&[], (1.0, 1.0)).tokens.is_empty());
    }

    #[test]
    fn test_match_ocr_to_cells_prefers_iou_then_distance() {
        let cells = vec![
            BoundingBox::new(0.0, 0.0, 50.0, 20.0),
            BoundingBox::new(50.0, 0.0, 100.0, 20.0),
        ];
        let ocr = vec![
            BoundingBox::new(55.0, 2.0, 90.0, 18.0),
            BoundingBox::new(5.0, 2.0, 40.0, 18.0),
            BoundingBox::new(200.0, 50.0, 210.0, 60.0), // no overlap -> nearest by distance
        ];
        assert_eq!(match_ocr_to_cells(&cells, &ocr), vec![vec![1], vec![0, 2]]);
        assert_eq!(match_ocr_to_cells(&[], &ocr), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn test_fill_html_single_and_multi_text() {
        let tokens = toks(&[
            "<tr>",
            "<td>",
            "</td>",
            "<td></td>",
            "<td",
            " colspan=\"2\"",
            ">",
            "</td>",
            "</tr>",
        ]);
        let texts = toks(&["Name", " A", "B", "x"]);
        let matched = vec![vec![0], vec![1, 2], vec![]];
        assert_eq!(
            fill_html(&tokens, &matched, &texts),
            "<html><body><table><tr><td>Name</td><td>A B</td><td colspan=\"2\"></td></tr></table></body></html>"
        );
    }

    #[test]
    fn test_fill_html_bold_handling() {
        let tokens = toks(&["<tr>", "<td></td>", "</tr>"]);
        let texts = toks(&["<b>Head", "er</b>"]);
        assert_eq!(
            fill_html(&tokens, &[vec![0, 1]], &texts),
            "<html><body><table><tr><td><b>Head er</b></td></tr></table></body></html>"
        );
    }

    #[test]
    fn test_cell_nms_and_row_sort() {
        let cells = vec![
            (BoundingBox::new(0.0, 0.0, 10.0, 10.0), 0.5),
            (BoundingBox::new(1.0, 1.0, 10.0, 10.0), 0.9),
            (BoundingBox::new(20.0, 0.0, 30.0, 10.0), 0.4),
        ];
        let kept = cell_nms(cells, 0.3);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].1, 0.9);

        let unsorted = vec![
            BoundingBox::new(50.0, 22.0, 90.0, 40.0),
            BoundingBox::new(50.0, 0.0, 90.0, 20.0),
            BoundingBox::new(0.0, 3.0, 50.0, 20.0),
            BoundingBox::new(0.0, 25.0, 50.0, 40.0),
        ];
        let sorted = sort_cells_row_major(&unsorted);
        let xy: Vec<_> = sorted.iter().map(|b| (b.x1, b.y1)).collect();
        assert_eq!(xy, vec![(0.0, 3.0), (50.0, 0.0), (0.0, 25.0), (50.0, 22.0)]);
    }
}
