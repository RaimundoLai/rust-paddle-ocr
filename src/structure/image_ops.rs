//! Image-to-tensor helpers for the document structure models.

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use ndarray::Array4;

/// Channel order written into the tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    /// Red, green, blue
    Rgb,
    /// Blue, green, red (Paddle models trained on OpenCV images)
    Bgr,
}

/// Per-channel normalization: `out = (pixel * scale - mean[c]) / std[c]`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalization {
    /// Multiplier applied to the raw 0..255 value
    pub scale: f32,
    /// Mean per tensor channel
    pub mean: [f32; 3],
    /// Standard deviation per tensor channel
    pub std: [f32; 3],
}

impl Normalization {
    /// Scale to 0..1 without mean/std
    pub const UNIT: Normalization = Normalization {
        scale: 1.0 / 255.0,
        mean: [0.0; 3],
        std: [1.0; 3],
    };

    /// ImageNet mean/std applied to channels in tensor order
    pub const IMAGENET: Normalization = Normalization {
        scale: 1.0 / 255.0,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };
}

/// Round half to even (Python's `round`), used to match Paddle's resize arithmetic
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Resize an RGB image exactly to `width x height`
pub fn resize_rgb(image: &RgbImage, width: u32, height: u32, filter: FilterType) -> RgbImage {
    if image.width() == width && image.height() == height {
        return image.clone();
    }
    image::imageops::resize(image, width.max(1), height.max(1), filter)
}

/// Convert an image to an NCHW tensor of shape `[1, 3, pad_h, pad_w]`.
///
/// The image is placed at the top-left; the remaining area is filled with `pad_value`
/// (applied after normalization).
pub fn to_nchw_padded(
    image: &RgbImage,
    order: ChannelOrder,
    norm: &Normalization,
    pad_w: u32,
    pad_h: u32,
    pad_value: f32,
) -> Array4<f32> {
    let (w, h) = image.dimensions();
    let pad_w = pad_w.max(w) as usize;
    let pad_h = pad_h.max(h) as usize;
    let mut tensor = Array4::<f32>::from_elem((1, 3, pad_h, pad_w), pad_value);
    let channel_map: [usize; 3] = match order {
        ChannelOrder::Rgb => [0, 1, 2],
        ChannelOrder::Bgr => [2, 1, 0],
    };
    let alpha: [f32; 3] = std::array::from_fn(|c| norm.scale / norm.std[c]);
    let beta: [f32; 3] = std::array::from_fn(|c| -norm.mean[c] / norm.std[c]);
    for (x, y, pixel) in image.enumerate_pixels() {
        for c in 0..3 {
            let v = pixel.0[channel_map[c]] as f32;
            tensor[[0, c, y as usize, x as usize]] = v * alpha[c] + beta[c];
        }
    }
    tensor
}

/// Convert an image to an NCHW tensor of shape `[1, 3, h, w]`
pub fn to_nchw(image: &RgbImage, order: ChannelOrder, norm: &Normalization) -> Array4<f32> {
    to_nchw_padded(image, order, norm, image.width(), image.height(), 0.0)
}

/// Crop a rectangle from an image (clipped to the image bounds)
pub fn crop_rgb(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> RgbImage {
    let x = x.min(image.width());
    let y = y.min(image.height());
    let width = width.min(image.width() - x).max(1);
    let height = height.min(image.height() - y).max(1);
    image::imageops::crop_imm(image, x, y, width, height).to_image()
}

/// Convert any image to RGB8
pub fn to_rgb8(image: &DynamicImage) -> RgbImage {
    image.to_rgb8()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn test_round_half_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(2.6), 3.0);
        assert_eq!(round_half_even(-1.5), -2.0);
    }

    #[test]
    fn test_to_nchw_channel_order_and_norm() {
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([255, 0, 51]));
        img.put_pixel(1, 0, Rgb([0, 255, 0]));
        let rgb = to_nchw(&img, ChannelOrder::Rgb, &Normalization::UNIT);
        assert_eq!(rgb.shape(), &[1, 3, 1, 2]);
        assert!((rgb[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((rgb[[0, 2, 0, 0]] - 0.2).abs() < 1e-6);
        let bgr = to_nchw(&img, ChannelOrder::Bgr, &Normalization::UNIT);
        assert!((bgr[[0, 0, 0, 0]] - 0.2).abs() < 1e-6);
        assert!((bgr[[0, 2, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_nchw_padded_fills_after_normalization() {
        let img = RgbImage::from_pixel(1, 1, Rgb([255, 255, 255]));
        let t = to_nchw_padded(&img, ChannelOrder::Rgb, &Normalization::IMAGENET, 3, 2, 0.0);
        assert_eq!(t.shape(), &[1, 3, 2, 3]);
        assert!((t[[0, 0, 0, 0]] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert_eq!(t[[0, 0, 1, 2]], 0.0);
    }

    #[test]
    fn test_crop_rgb_clips() {
        let img = RgbImage::new(10, 8);
        let c = crop_rgb(&img, 8, 6, 5, 5);
        assert_eq!(c.dimensions(), (2, 2));
    }
}
