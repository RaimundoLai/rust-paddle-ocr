//! Document image unwarping with UVDoc.
//!
//! The exported model predicts a sampling mesh internally and resamples the full
//! resolution input, so the output is the rectified image at the input size.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};

use super::image_ops::{to_nchw, to_rgb8, ChannelOrder, Normalization};
use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, MultiModel};

/// UVDoc unwarping model
pub struct DocUnwarpModel {
    model: MultiModel,
}

impl DocUnwarpModel {
    /// Load from a file
    pub fn from_file(path: impl AsRef<Path>, config: Option<InferenceConfig>) -> OcrResult<Self> {
        Self::from_bytes(&std::fs::read(path.as_ref())?, config)
    }

    /// Load from bytes
    pub fn from_bytes(bytes: &[u8], config: Option<InferenceConfig>) -> OcrResult<Self> {
        Ok(Self {
            model: MultiModel::from_buffer(bytes, &["image"], &["fetch_name_0"], config)?,
        })
    }

    /// Rectify a photographed or curved document page
    pub fn unwarp(&self, image: &DynamicImage) -> OcrResult<DynamicImage> {
        let rgb = to_rgb8(image);
        let (w, h) = rgb.dimensions();
        if w == 0 || h == 0 {
            return Err(OcrError::InvalidParameter("Image is empty".to_string()));
        }
        let input = to_nchw(&rgb, ChannelOrder::Bgr, &Normalization::UNIT);
        let out = self.model.run(&[input.view().into_dyn()])?;
        let tensor = out.first().ok_or_else(|| {
            OcrError::PostprocessError("Unwarp model returned no output".to_string())
        })?;
        if tensor.shape.len() != 4 || tensor.shape[1] != 3 {
            return Err(OcrError::PostprocessError(format!(
                "Unexpected unwarp output shape {:?}",
                tensor.shape
            )));
        }
        let (oh, ow) = (tensor.shape[2], tensor.shape[3]);
        let data = tensor.to_f32_vec();
        Ok(DynamicImage::ImageRgb8(bgr_planes_to_rgb(&data, ow, oh)))
    }
}

/// Convert planar BGR floats in `[0, 1]` to an RGB image
pub(crate) fn bgr_planes_to_rgb(data: &[f32], width: usize, height: usize) -> RgbImage {
    let plane = width * height;
    let to_u8 = |v: f32| (v * 255.0).clamp(0.0, 255.0) as u8;
    RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let i = y as usize * width + x as usize;
        Rgb([
            to_u8(data[2 * plane + i]),
            to_u8(data[plane + i]),
            to_u8(data[i]),
        ])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgr_planes_to_rgb() {
        // 2x1 image: planes B, G, R
        let data = [0.0, 1.0, 0.5, 0.0, 1.0, 2.0];
        let img = bgr_planes_to_rgb(&data, 2, 1);
        assert_eq!(img.get_pixel(0, 0).0, [255, 127, 0]);
        assert_eq!(img.get_pixel(1, 0).0, [255, 0, 255]);
    }
}
