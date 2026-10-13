//! Axis-aligned box geometry shared by the document structure modules.

/// Axis-aligned bounding box in pixel coordinates (`x2`/`y2` exclusive edges)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoundingBox {
    /// Left edge
    pub x1: f32,
    /// Top edge
    pub y1: f32,
    /// Right edge
    pub x2: f32,
    /// Bottom edge
    pub y2: f32,
}

impl BoundingBox {
    /// Create a box from its edges
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Box width (0 when degenerate)
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    /// Box height (0 when degenerate)
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    /// Box area
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Whether the box has a positive area
    pub fn is_valid(&self) -> bool {
        self.x2 > self.x1 && self.y2 > self.y1
    }

    /// Box center
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// Intersection area with another box
    pub fn intersection_area(&self, other: &BoundingBox) -> f32 {
        let w = self.x2.min(other.x2) - self.x1.max(other.x1);
        let h = self.y2.min(other.y2) - self.y1.max(other.y1);
        if w <= 0.0 || h <= 0.0 {
            0.0
        } else {
            w * h
        }
    }

    /// Intersection over union
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let inter = self.intersection_area(other);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// IoU with PaddleDetection's inclusive-pixel (+1) convention, used by layout NMS
    pub fn iou_inclusive(&self, other: &BoundingBox) -> f32 {
        let iw = (self.x2.min(other.x2) - self.x1.max(other.x1) + 1.0).max(0.0);
        let ih = (self.y2.min(other.y2) - self.y1.max(other.y1) + 1.0).max(0.0);
        let inter = iw * ih;
        let a = (self.x2 - self.x1 + 1.0) * (self.y2 - self.y1 + 1.0);
        let b = (other.x2 - other.x1 + 1.0) * (other.y2 - other.y1 + 1.0);
        let union = a + b - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Fraction of this box covered by `other` (intersection / own area)
    pub fn covered_ratio(&self, other: &BoundingBox) -> f32 {
        let area = self.area();
        if area <= 0.0 {
            0.0
        } else {
            self.intersection_area(other) / area
        }
    }

    /// Intersection divided by the smaller of the two areas
    pub fn overlap_ratio_small(&self, other: &BoundingBox) -> f32 {
        let min_area = self.area().min(other.area());
        if min_area <= 0.0 {
            0.0
        } else {
            self.intersection_area(other) / min_area
        }
    }

    /// Smallest box containing both boxes
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox::new(
            self.x1.min(other.x1),
            self.y1.min(other.y1),
            self.x2.max(other.x2),
            self.y2.max(other.y2),
        )
    }

    /// Clip the box to `[0, width] x [0, height]`
    pub fn clip(&self, width: f32, height: f32) -> BoundingBox {
        BoundingBox::new(
            self.x1.clamp(0.0, width),
            self.y1.clamp(0.0, height),
            self.x2.clamp(0.0, width),
            self.y2.clamp(0.0, height),
        )
    }

    /// Translate the box
    pub fn offset(&self, dx: f32, dy: f32) -> BoundingBox {
        BoundingBox::new(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)
    }

    /// Integer pixel rectangle `(x, y, width, height)` clipped to the image, if non-empty
    pub fn to_pixel_rect(
        &self,
        image_width: u32,
        image_height: u32,
    ) -> Option<(u32, u32, u32, u32)> {
        let x1 = self.x1.max(0.0).floor() as u32;
        let y1 = self.y1.max(0.0).floor() as u32;
        let x2 = (self.x2.min(image_width as f32).ceil().max(0.0) as u32).min(image_width);
        let y2 = (self.y2.min(image_height as f32).ceil().max(0.0) as u32).min(image_height);
        if x2 > x1 && y2 > y1 {
            Some((x1, y1, x2 - x1, y2 - y1))
        } else {
            None
        }
    }

    /// Box enclosing a set of points
    pub fn from_points(points: &[[f32; 2]]) -> Option<BoundingBox> {
        let first = points.first()?;
        let mut b = BoundingBox::new(first[0], first[1], first[0], first[1]);
        for p in &points[1..] {
            b.x1 = b.x1.min(p[0]);
            b.y1 = b.y1.min(p[1]);
            b.x2 = b.x2.max(p[0]);
            b.y2 = b.y2.max(p[1]);
        }
        Some(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area_and_iou() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        assert_eq!(a.area(), 100.0);
        assert_eq!(a.intersection_area(&b), 25.0);
        assert!((a.iou(&b) - 25.0 / 175.0).abs() < 1e-6);
        assert_eq!(a.iou(&BoundingBox::new(20.0, 20.0, 30.0, 30.0)), 0.0);
    }

    #[test]
    fn test_degenerate_boxes() {
        let d = BoundingBox::new(5.0, 5.0, 5.0, 9.0);
        assert!(!d.is_valid());
        assert_eq!(d.area(), 0.0);
        assert_eq!(
            d.covered_ratio(&BoundingBox::new(0.0, 0.0, 10.0, 10.0)),
            0.0
        );
        assert_eq!(
            d.overlap_ratio_small(&BoundingBox::new(0.0, 0.0, 10.0, 10.0)),
            0.0
        );
    }

    #[test]
    fn test_iou_inclusive_matches_paddle_convention() {
        let a = BoundingBox::new(0.0, 0.0, 9.0, 9.0);
        // Identical boxes are a perfect match, and touching boxes overlap by one pixel row.
        assert!((a.iou_inclusive(&a) - 1.0).abs() < 1e-6);
        let b = BoundingBox::new(9.0, 0.0, 18.0, 9.0);
        assert!((a.iou_inclusive(&b) - 10.0 / 190.0).abs() < 1e-6);
    }

    #[test]
    fn test_covered_ratio_and_union() {
        let inner = BoundingBox::new(2.0, 2.0, 4.0, 4.0);
        let outer = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(inner.covered_ratio(&outer), 1.0);
        assert!((outer.covered_ratio(&inner) - 0.04).abs() < 1e-6);
        assert_eq!(inner.union(&outer), outer);
    }

    #[test]
    fn test_clip_and_pixel_rect() {
        let b = BoundingBox::new(-3.0, 2.5, 120.2, 50.0);
        assert_eq!(b.clip(100.0, 40.0), BoundingBox::new(0.0, 2.5, 100.0, 40.0));
        assert_eq!(b.to_pixel_rect(100, 40), Some((0, 2, 100, 38)));
        assert_eq!(
            BoundingBox::new(150.0, 0.0, 160.0, 10.0).to_pixel_rect(100, 40),
            None
        );
    }

    #[test]
    fn test_from_points() {
        assert_eq!(BoundingBox::from_points(&[]), None);
        let b = BoundingBox::from_points(&[[3.0, 4.0], [1.0, 8.0], [5.0, 2.0]]).unwrap();
        assert_eq!(b, BoundingBox::new(1.0, 2.0, 5.0, 8.0));
    }
}
