//! Shared runner for Paddle-exported RT-DETR detectors (layout and table cell models).
//!
//! These models take `im_shape`, `image` and `scale_factor`, and return rows of
//! `[class, score, x1, y1, x2, y2(, order)]` already mapped to original pixels.

use image::imageops::FilterType;
use image::RgbImage;
use ndarray::{arr2, ArrayViewD};

use super::geometry::BoundingBox;
use super::image_ops::{resize_rgb, to_nchw, ChannelOrder, Normalization};
use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, MultiModel, Tensor};

/// Input tensor names of Paddle DETR exports
pub(crate) const DETR_INPUTS: [&str; 3] = ["im_shape", "image", "scale_factor"];

/// One raw detection row
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DetrRow {
    pub class_id: usize,
    pub score: f32,
    pub bbox: BoundingBox,
    /// Reading-order rank predicted by the model (PP-DocLayoutV3 only)
    pub order: Option<f32>,
    /// Row index in the raw output, used to look up masks
    pub index: usize,
}

/// Binary instance masks `[count, height, width]` over the resized network input
#[derive(Debug, Clone)]
pub(crate) struct DetrMasks {
    pub data: Vec<i32>,
    pub height: usize,
    pub width: usize,
}

impl DetrMasks {
    pub fn get(&self, index: usize, x: usize, y: usize) -> i32 {
        self.data[index * self.height * self.width + y * self.width + x]
    }
}

/// Raw detector output
#[derive(Debug, Clone)]
pub(crate) struct DetrOutput {
    pub rows: Vec<DetrRow>,
    pub masks: Option<DetrMasks>,
}

/// RT-DETR style detector
pub(crate) struct DetrModel {
    model: MultiModel,
    input_size: u32,
}

impl DetrModel {
    /// Load a detector. `num_outputs` is 2 (boxes, counts) or 3 (with masks).
    pub fn from_bytes(
        bytes: &[u8],
        num_outputs: usize,
        input_size: u32,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let outputs: Vec<String> = (0..num_outputs)
            .map(|i| format!("fetch_name_{}", i))
            .collect();
        let output_refs: Vec<&str> = outputs.iter().map(|s| s.as_str()).collect();
        let model = MultiModel::from_buffer(bytes, &DETR_INPUTS, &output_refs, config)?;
        Ok(Self { model, input_size })
    }

    /// Run detection on an RGB image
    pub fn detect(&self, image: &RgbImage) -> OcrResult<DetrOutput> {
        let (w, h) = image.dimensions();
        if w == 0 || h == 0 {
            return Err(OcrError::InvalidParameter("Image is empty".to_string()));
        }
        let size = self.input_size;
        let resized = resize_rgb(image, size, size, FilterType::CatmullRom);
        let pixels = to_nchw(&resized, ChannelOrder::Rgb, &Normalization::UNIT);
        let s = size as f32;
        let im_shape = arr2(&[[s, s]]).into_dyn();
        let scale_factor = arr2(&[[s / h as f32, s / w as f32]]).into_dyn();
        let inputs: [ArrayViewD<f32>; 3] = [
            im_shape.view(),
            pixels.view().into_dyn(),
            scale_factor.view(),
        ];
        let outputs = self.model.run(&inputs)?;
        parse_detr_outputs(&outputs)
    }
}

/// Parse `[boxes, counts(, masks)]` outputs of a single-image DETR run
pub(crate) fn parse_detr_outputs(outputs: &[Tensor]) -> OcrResult<DetrOutput> {
    let boxes = outputs
        .first()
        .ok_or_else(|| OcrError::PostprocessError("Detector returned no outputs".to_string()))?;
    let cols = *boxes.shape.last().unwrap_or(&0);
    if cols < 6 || boxes.shape.len() != 2 {
        return Err(OcrError::PostprocessError(format!(
            "Unexpected detector output shape {:?}",
            boxes.shape
        )));
    }
    let data = boxes.to_f32_vec();
    let mut num_rows = boxes.shape[0];
    if let Some(count) = outputs.get(1).and_then(|t| t.to_f32_vec().first().copied()) {
        num_rows = num_rows.min(count.max(0.0) as usize);
    }

    let rows = (0..num_rows)
        .filter_map(|i| {
            let r = &data[i * cols..(i + 1) * cols];
            if r[0] < 0.0 || !r[1].is_finite() {
                return None;
            }
            Some(DetrRow {
                class_id: r[0] as usize,
                score: r[1],
                bbox: BoundingBox::new(r[2], r[3], r[4], r[5]),
                order: if cols >= 7 { Some(r[6]) } else { None },
                index: i,
            })
        })
        .collect();

    let masks = match outputs.get(2) {
        Some(t) if t.shape.len() == 3 => Some(DetrMasks {
            data: t.as_i32().map(|v| v.to_vec()).unwrap_or_else(|| {
                t.to_f32_vec()
                    .into_iter()
                    .map(|v| (v > 0.5) as i32)
                    .collect()
            }),
            height: t.shape[1],
            width: t.shape[2],
        }),
        _ => None,
    };
    Ok(DetrOutput { rows, masks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnn::TensorData;

    #[test]
    fn test_parse_detr_outputs_with_order_and_count() {
        let boxes = Tensor {
            shape: vec![3, 7],
            data: TensorData::F32(vec![
                2.0, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, //
                -1.0, 0.8, 0.0, 0.0, 1.0, 1.0, 0.0, //
                1.0, 0.7, 5.0, 6.0, 7.0, 8.0, 1.0,
            ]),
        };
        let count = Tensor {
            shape: vec![1],
            data: TensorData::I32(vec![3]),
        };
        let out = parse_detr_outputs(&[boxes, count]).unwrap();
        assert_eq!(out.rows.len(), 2);
        assert_eq!(out.rows[0].class_id, 2);
        assert_eq!(out.rows[0].order, Some(5.0));
        assert_eq!(out.rows[1].index, 2);
        assert!(out.masks.is_none());
    }

    #[test]
    fn test_parse_detr_outputs_rejects_bad_shape() {
        let boxes = Tensor {
            shape: vec![2, 4],
            data: TensorData::F32(vec![0.0; 8]),
        };
        assert!(parse_detr_outputs(&[boxes]).is_err());
        assert!(parse_detr_outputs(&[]).is_err());
    }

    #[test]
    fn test_parse_detr_outputs_masks() {
        let boxes = Tensor {
            shape: vec![1, 6],
            data: TensorData::F32(vec![0.0, 0.5, 0.0, 0.0, 2.0, 2.0]),
        };
        let count = Tensor {
            shape: vec![1],
            data: TensorData::I32(vec![1]),
        };
        let masks = Tensor {
            shape: vec![1, 2, 2],
            data: TensorData::I32(vec![0, 1, 1, 0]),
        };
        let out = parse_detr_outputs(&[boxes, count, masks]).unwrap();
        let m = out.masks.unwrap();
        assert_eq!((m.height, m.width), (2, 2));
        assert_eq!(m.get(0, 1, 0), 1);
        assert_eq!(m.get(0, 1, 1), 0);
    }
}
