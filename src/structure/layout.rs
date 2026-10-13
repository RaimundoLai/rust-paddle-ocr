//! Document layout detection (PP-DocLayoutV3 and PP-DocLayout_plus-L).

use std::path::Path;

use image::{DynamicImage, GrayImage, Luma};

use super::detr::{DetrMasks, DetrModel, DetrRow};
use super::geometry::BoundingBox;
use super::image_ops::{round_half_even, to_rgb8};
use super::polygon::{mask_to_polygon, polygon_area};
use super::reading_order::xycut_order;
use crate::error::{OcrError, OcrResult};
use crate::mnn::InferenceConfig;

/// Labels of PP-DocLayoutV3 (25 classes)
pub const DOC_LAYOUT_V3_LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

/// Labels of PP-DocLayout_plus-L (20 classes)
pub const DOC_LAYOUT_PLUS_L_LABELS: [&str; 20] = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart",
    "formula_number",
    "aside_text",
    "reference_content",
];

/// Labels that do not receive a reading-order number
const SKIP_ORDER_LABELS: [&str; 12] = [
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
    "number",
];

/// Supported layout models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutModelType {
    /// PP-DocLayoutV3: 25 classes, predicted reading order and instance masks
    DocLayoutV3,
    /// PP-DocLayout_plus-L: 20 classes, reading order computed by XY-cut
    DocLayoutPlusL,
}

impl LayoutModelType {
    /// Class labels in model output order
    pub fn labels(self) -> &'static [&'static str] {
        match self {
            LayoutModelType::DocLayoutV3 => &DOC_LAYOUT_V3_LABELS,
            LayoutModelType::DocLayoutPlusL => &DOC_LAYOUT_PLUS_L_LABELS,
        }
    }

    fn num_outputs(self) -> usize {
        match self {
            LayoutModelType::DocLayoutV3 => 3,
            LayoutModelType::DocLayoutPlusL => 2,
        }
    }

    fn class_id(self, label: &str) -> Option<usize> {
        self.labels().iter().position(|l| *l == label)
    }
}

/// Layout post-processing options
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutOptions {
    /// Default score threshold (exclusive)
    pub score_threshold: f32,
    /// Per-class score thresholds overriding the default
    pub class_thresholds: Vec<(usize, f32)>,
    /// Apply class-aware NMS
    pub nms: bool,
    /// Classes whose boxes absorb boxes mostly contained in them ("large" merge mode)
    pub merge_large_classes: Vec<usize>,
    /// Drop heavily overlapping and tiny boxes (PaddleOCR-VL `filter_overlap_boxes`)
    pub filter_overlaps: bool,
    /// Extract polygons from instance masks (PP-DocLayoutV3 only)
    pub with_polygons: bool,
}

impl LayoutOptions {
    /// Defaults used by PaddleOCR for the given model
    pub fn for_model(model_type: LayoutModelType) -> Self {
        match model_type {
            // PaddleOCR-VL-1.5 settings.
            LayoutModelType::DocLayoutV3 => Self {
                score_threshold: 0.3,
                class_thresholds: Vec::new(),
                nms: true,
                merge_large_classes: vec![3, 5, 6, 15, 17],
                filter_overlaps: true,
                with_polygons: true,
            },
            // PP-StructureV3 settings.
            LayoutModelType::DocLayoutPlusL => Self {
                score_threshold: 0.5,
                class_thresholds: vec![(0, 0.3), (2, 0.4), (7, 0.3), (15, 0.45)],
                nms: true,
                merge_large_classes: vec![0, 1, 7, 16],
                filter_overlaps: false,
                with_polygons: false,
            },
        }
    }

    /// Set the default score threshold
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = threshold;
        self
    }

    /// Enable or disable polygon extraction
    pub fn with_polygons(mut self, enable: bool) -> Self {
        self.with_polygons = enable;
        self
    }

    /// Enable or disable overlap filtering
    pub fn with_filter_overlaps(mut self, enable: bool) -> Self {
        self.filter_overlaps = enable;
        self
    }

    fn threshold_for(&self, class_id: usize) -> f32 {
        self.class_thresholds
            .iter()
            .find(|(c, _)| *c == class_id)
            .map(|(_, t)| *t)
            .unwrap_or(self.score_threshold)
    }
}

/// A detected layout region
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutRegion {
    /// Class index in the model label list
    pub class_id: usize,
    /// Class label
    pub label: &'static str,
    /// Detection score
    pub score: f32,
    /// Bounding box clipped to the image
    pub bbox: BoundingBox,
    /// Polygon outline when the region is not well described by its box
    pub polygon: Option<Vec<[f32; 2]>>,
    /// 1-based reading-order number; `None` for labels outside the main text flow
    pub order: Option<usize>,
}

/// Layout detection model
pub struct LayoutModel {
    detector: DetrModel,
    model_type: LayoutModelType,
    options: LayoutOptions,
}

impl LayoutModel {
    /// Load a layout model from a file
    pub fn from_file(
        model_path: impl AsRef<Path>,
        model_type: LayoutModelType,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let bytes = std::fs::read(model_path.as_ref())
            .map_err(|e| OcrError::ModelLoadError(format!("Failed to read layout model: {}", e)))?;
        Self::from_bytes(&bytes, model_type, config)
    }

    /// Load a layout model from bytes
    pub fn from_bytes(
        bytes: &[u8],
        model_type: LayoutModelType,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let detector = DetrModel::from_bytes(bytes, model_type.num_outputs(), 800, config)?;
        Ok(Self {
            detector,
            model_type,
            options: LayoutOptions::for_model(model_type),
        })
    }

    /// Replace the post-processing options
    pub fn with_options(mut self, options: LayoutOptions) -> Self {
        self.options = options;
        self
    }

    /// Current options
    pub fn options(&self) -> &LayoutOptions {
        &self.options
    }

    /// Model type
    pub fn model_type(&self) -> LayoutModelType {
        self.model_type
    }

    /// Detect layout regions, returned in reading order
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<LayoutRegion>> {
        let rgb = to_rgb8(image);
        let output = self.detector.detect(&rgb)?;
        Ok(postprocess_layout(
            output.rows,
            output.masks.as_ref(),
            self.model_type,
            &self.options,
            rgb.width() as f32,
            rgb.height() as f32,
        ))
    }
}

/// Greedy class-aware NMS used by PaddleX layout post-processing.
/// Returns indices in descending score order.
pub(crate) fn layout_nms(rows: &[DetrRow], iou_same: f32, iou_diff: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        rows[b]
            .score
            .partial_cmp(&rows[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    while let Some((&cur, rest)) = order.split_first() {
        keep.push(cur);
        order = rest
            .iter()
            .copied()
            .filter(|&i| {
                let limit = if rows[i].class_id == rows[cur].class_id {
                    iou_same
                } else {
                    iou_diff
                };
                rows[cur].bbox.iou_inclusive(&rows[i].bbox) < limit
            })
            .collect();
    }
    keep
}

/// Remove boxes at least 90% covered by a box of one of `classes`
pub(crate) fn merge_large(
    rows: &[DetrRow],
    classes: &[usize],
    formula_class: Option<usize>,
) -> Vec<bool> {
    let mut keep = vec![true; rows.len()];
    for &cat in classes {
        for i in 0..rows.len() {
            for j in 0..rows.len() {
                if i == j || rows[j].class_id != cat {
                    continue;
                }
                if let Some(f) = formula_class {
                    if rows[i].class_id == f && rows[j].class_id != f {
                        continue;
                    }
                }
                if rows[i].bbox.covered_ratio(&rows[j].bbox) >= 0.9 {
                    keep[i] = false;
                }
            }
        }
    }
    keep
}

/// Drop full-page "image" detections that cover most of the page
pub(crate) fn filter_large_images(
    rows: Vec<DetrRow>,
    image_class: Option<usize>,
    width: f32,
    height: f32,
) -> Vec<DetrRow> {
    let Some(image_class) = image_class else {
        return rows;
    };
    if rows.len() <= 1 {
        return rows;
    }
    let ratio = if width > height { 0.82 } else { 0.93 };
    let limit = ratio * width * height;
    let filtered: Vec<DetrRow> = rows
        .iter()
        .filter(|r| r.class_id != image_class || r.bbox.clip(width, height).area() <= limit)
        .cloned()
        .collect();
    if filtered.is_empty() {
        rows
    } else {
        filtered
    }
}

/// PaddleOCR-VL `filter_overlap_boxes`: drop references, tiny boxes and duplicates
pub(crate) fn filter_overlap_regions(regions: Vec<LayoutRegion>) -> Vec<LayoutRegion> {
    let regions: Vec<LayoutRegion> = regions
        .into_iter()
        .filter(|r| r.label != "reference")
        .collect();
    let n = regions.len();
    let mut dropped = vec![false; n];
    let vision = ["image", "table", "seal", "chart"];
    for i in 0..n {
        let bi = &regions[i].bbox;
        if bi.width() < 6.0 || bi.height() < 6.0 {
            dropped[i] = true;
        }
        for j in (i + 1)..n {
            if dropped[i] || dropped[j] {
                continue;
            }
            let (ri, rj) = (&regions[i], &regions[j]);
            let ratio = ri.bbox.overlap_ratio_small(&rj.bbox);
            if ri.label == "inline_formula" || rj.label == "inline_formula" {
                if ratio > 0.5 {
                    if ri.label == "inline_formula" {
                        dropped[i] = true;
                    }
                    if rj.label == "inline_formula" {
                        dropped[j] = true;
                    }
                }
                continue;
            }
            if ratio > 0.7 {
                if ri.polygon.is_some() || rj.polygon.is_some() {
                    let pi = region_outline(ri);
                    let pj = region_outline(rj);
                    if polygon_overlap_small(&pi, &pj) < 0.7 {
                        continue;
                    }
                }
                let labels_differ = ri.label != rj.label;
                let any_vision = vision.contains(&ri.label) || vision.contains(&rj.label);
                if any_vision && labels_differ {
                    let has_table = ri.label == "table" || rj.label == "table";
                    let all_vision = vision.contains(&ri.label) && vision.contains(&rj.label);
                    if !has_table || all_vision {
                        continue;
                    }
                }
                if ri.bbox.area() >= rj.bbox.area() {
                    dropped[j] = true;
                } else {
                    dropped[i] = true;
                }
            }
        }
    }
    regions
        .into_iter()
        .zip(dropped)
        .filter(|(_, d)| !d)
        .map(|(r, _)| r)
        .collect()
}

fn region_outline(region: &LayoutRegion) -> Vec<[f32; 2]> {
    region.polygon.clone().unwrap_or_else(|| {
        let b = &region.bbox;
        vec![[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]]
    })
}

fn point_in_polygon(x: f32, y: f32, poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let ([xi, yi], [xj, yj]) = (poly[i], poly[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Intersection over the smaller polygon area, estimated on a sampling grid
pub(crate) fn polygon_overlap_small(a: &[[f32; 2]], b: &[[f32; 2]]) -> f32 {
    let (Some(ba), Some(bb)) = (BoundingBox::from_points(a), BoundingBox::from_points(b)) else {
        return 0.0;
    };
    let min_area = polygon_area(a).min(polygon_area(b));
    let x1 = ba.x1.max(bb.x1);
    let y1 = ba.y1.max(bb.y1);
    let x2 = ba.x2.min(bb.x2);
    let y2 = ba.y2.min(bb.y2);
    if min_area <= 0.0 || x2 <= x1 || y2 <= y1 {
        return 0.0;
    }
    const STEPS: usize = 128;
    let (dx, dy) = ((x2 - x1) / STEPS as f32, (y2 - y1) / STEPS as f32);
    let mut hits = 0usize;
    for iy in 0..STEPS {
        let y = y1 + (iy as f32 + 0.5) * dy;
        for ix in 0..STEPS {
            let x = x1 + (ix as f32 + 0.5) * dx;
            if point_in_polygon(x, y, a) && point_in_polygon(x, y, b) {
                hits += 1;
            }
        }
    }
    (hits as f32 * dx * dy / min_area).min(1.0)
}

/// Polygon for one detection from the V3 mask, in image coordinates
fn region_polygon(
    masks: &DetrMasks,
    index: usize,
    bbox: &BoundingBox,
    width: f32,
    height: f32,
) -> Option<Vec<[f32; 2]>> {
    let (x1, y1, x2, y2) = (
        bbox.x1 as i64,
        bbox.y1 as i64,
        bbox.x2 as i64,
        bbox.y2 as i64,
    );
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let sx = masks.width as f32 / width;
    let sy = masks.height as f32 / height;
    let mx1 = (round_half_even((x1 as f32 * sx) as f64) as i64).clamp(0, masks.width as i64) as u32;
    let mx2 = (round_half_even((x2 as f32 * sx) as f64) as i64).clamp(0, masks.width as i64) as u32;
    let my1 =
        (round_half_even((y1 as f32 * sy) as f64) as i64).clamp(0, masks.height as i64) as u32;
    let my2 =
        (round_half_even((y2 as f32 * sy) as f64) as i64).clamp(0, masks.height as i64) as u32;
    if mx2 <= mx1 || my2 <= my1 {
        return None;
    }
    // One pixel of background padding keeps border-touching regions as closed contours.
    let (cw, ch) = (mx2 - mx1, my2 - my1);
    let mut crop = GrayImage::new(cw + 2, ch + 2);
    let mut any = false;
    for y in 0..ch {
        for x in 0..cw {
            if masks.get(index, (mx1 + x) as usize, (my1 + y) as usize) > 0 {
                crop.put_pixel(x + 1, y + 1, Luma([255]));
                any = true;
            }
        }
    }
    if !any {
        return None;
    }
    let poly = mask_to_polygon(&crop, 0.004)?;
    let (bw, bh) = ((x2 - x1) as f32, (y2 - y1) as f32);
    // Contour points are pixel centres; stretch the outermost pixels onto the box edges.
    let (kx, ky) = (bw / (cw.max(2) - 1) as f32, bh / (ch.max(2) - 1) as f32);
    let points: Vec<[f32; 2]> = poly
        .iter()
        .map(|p| {
            [
                ((p[0] - 1.0) * kx + x1 as f32).clamp(0.0, width),
                ((p[1] - 1.0) * ky + y1 as f32).clamp(0.0, height),
            ]
        })
        .collect();
    // Near-rectangular outlines add nothing over the box.
    let rect_area = bw * bh;
    if points.len() < 4
        || (rect_area > 0.0 && polygon_area(&points) >= 0.95 * rect_area && points.len() <= 4)
    {
        return None;
    }
    Some(points)
}

/// Full layout post-processing: threshold, NMS, merge, order, polygons, clipping
pub(crate) fn postprocess_layout(
    rows: Vec<DetrRow>,
    masks: Option<&DetrMasks>,
    model_type: LayoutModelType,
    options: &LayoutOptions,
    width: f32,
    height: f32,
) -> Vec<LayoutRegion> {
    let labels = model_type.labels();
    let mut rows: Vec<DetrRow> = rows
        .into_iter()
        .filter(|r| r.class_id < labels.len())
        .map(|mut r| {
            if model_type == LayoutModelType::DocLayoutV3 {
                let rd = |v: f32| round_half_even(v as f64) as f32;
                r.bbox =
                    BoundingBox::new(rd(r.bbox.x1), rd(r.bbox.y1), rd(r.bbox.x2), rd(r.bbox.y2));
            }
            r
        })
        .filter(|r| r.score > options.threshold_for(r.class_id))
        .collect();

    if options.nms {
        let keep = layout_nms(&rows, 0.6, 0.98);
        rows = keep.into_iter().map(|i| rows[i].clone()).collect();
    }
    rows = filter_large_images(rows, model_type.class_id("image"), width, height);
    if !options.merge_large_classes.is_empty() {
        let keep = merge_large(
            &rows,
            &options.merge_large_classes,
            model_type.class_id("formula"),
        );
        rows = rows
            .into_iter()
            .zip(keep)
            .filter(|(_, k)| *k)
            .map(|(r, _)| r)
            .collect();
    }

    // Reading order: model rank when available, XY-cut otherwise.
    if rows.iter().all(|r| r.order.is_some()) && !rows.is_empty() {
        rows.sort_by(|a, b| {
            a.order
                .partial_cmp(&b.order)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        let boxes: Vec<BoundingBox> = rows.iter().map(|r| r.bbox.clip(width, height)).collect();
        let order = xycut_order(&boxes);
        rows = order.into_iter().map(|i| rows[i].clone()).collect();
    }

    let mut regions: Vec<LayoutRegion> = rows
        .iter()
        .filter_map(|r| {
            let polygon = match (options.with_polygons, masks) {
                (true, Some(m)) => region_polygon(m, r.index, &r.bbox, width, height),
                _ => None,
            };
            // `+ 0.0` turns a negative zero from rounding into a positive zero.
            let clipped = BoundingBox::new(
                r.bbox.x1.max(0.0).trunc() + 0.0,
                r.bbox.y1.max(0.0).trunc() + 0.0,
                r.bbox.x2.min(width).trunc() + 0.0,
                r.bbox.y2.min(height).trunc() + 0.0,
            );
            if !clipped.is_valid() {
                return None;
            }
            Some(LayoutRegion {
                class_id: r.class_id,
                label: labels[r.class_id],
                score: r.score,
                bbox: clipped,
                polygon,
                order: None,
            })
        })
        .collect();

    if options.filter_overlaps {
        regions = filter_overlap_regions(regions);
    }
    let mut next = 1;
    for r in regions.iter_mut() {
        if !SKIP_ORDER_LABELS.contains(&r.label) {
            r.order = Some(next);
            next += 1;
        }
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(class_id: usize, score: f32, b: [f32; 4], order: Option<f32>) -> DetrRow {
        DetrRow {
            class_id,
            score,
            bbox: BoundingBox::new(b[0], b[1], b[2], b[3]),
            order,
            index: 0,
        }
    }

    #[test]
    fn test_layout_nms_class_aware() {
        let rows = vec![
            row(0, 0.9, [0.0, 0.0, 100.0, 100.0], None),
            row(0, 0.8, [5.0, 5.0, 100.0, 100.0], None), // same class, high IoU -> removed
            row(1, 0.7, [5.0, 5.0, 100.0, 100.0], None), // other class, IoU < 0.98 -> kept
            row(1, 0.95, [300.0, 0.0, 400.0, 50.0], None), // disjoint
        ];
        assert_eq!(layout_nms(&rows, 0.6, 0.98), vec![3, 0, 2]);
    }

    #[test]
    fn test_merge_large_removes_contained_boxes() {
        let rows = vec![
            row(3, 0.9, [0.0, 0.0, 100.0, 100.0], None), // chart (merge class)
            row(22, 0.8, [10.0, 10.0, 50.0, 50.0], None), // text inside chart
            row(22, 0.8, [90.0, 90.0, 200.0, 200.0], None),
        ];
        assert_eq!(merge_large(&rows, &[3], None), vec![true, false, true]);
    }

    #[test]
    fn test_merge_large_formula_exemption() {
        let rows = vec![
            row(7, 0.9, [10.0, 10.0, 20.0, 20.0], None),
            row(2, 0.9, [0.0, 0.0, 100.0, 100.0], None),
        ];
        // The formula box is not removed by a text box even if text were a merge class.
        assert_eq!(merge_large(&rows, &[2], Some(7)), vec![true, true]);
    }

    #[test]
    fn test_filter_large_images_keeps_all_when_everything_dropped() {
        let full = row(14, 0.9, [0.0, 0.0, 100.0, 100.0], None);
        let other = row(22, 0.9, [0.0, 0.0, 10.0, 10.0], None);
        let kept = filter_large_images(vec![full.clone(), other.clone()], Some(14), 100.0, 100.0);
        assert_eq!(kept, vec![other]);
        let only_images = vec![full.clone(), full.clone()];
        assert_eq!(
            filter_large_images(only_images.clone(), Some(14), 100.0, 100.0),
            only_images
        );
    }

    #[test]
    fn test_postprocess_v3_orders_by_model_rank_and_numbers() {
        let rows = vec![
            row(22, 0.9, [0.0, 50.0, 100.0, 90.0], Some(2.0)), // text, second
            row(17, 0.9, [0.0, 0.0, 100.0, 20.0], Some(1.0)),  // paragraph_title, first
            row(12, 0.9, [0.0, 90.0, 100.0, 100.0], Some(0.0)), // header: no order number
            row(22, 0.1, [0.0, 0.0, 10.0, 10.0], Some(3.0)),   // below threshold
        ];
        let opts = LayoutOptions::for_model(LayoutModelType::DocLayoutV3).with_polygons(false);
        let regions = postprocess_layout(
            rows,
            None,
            LayoutModelType::DocLayoutV3,
            &opts,
            100.0,
            100.0,
        );
        let labels: Vec<_> = regions.iter().map(|r| r.label).collect();
        assert_eq!(labels, vec!["header", "paragraph_title", "text"]);
        let orders: Vec<_> = regions.iter().map(|r| r.order).collect();
        assert_eq!(orders, vec![None, Some(1), Some(2)]);
    }

    #[test]
    fn test_postprocess_plus_l_uses_class_thresholds_and_xycut() {
        let rows = vec![
            row(2, 0.45, [55.0, 30.0, 100.0, 100.0], None), // text (threshold 0.4) right column
            row(2, 0.45, [0.0, 30.0, 45.0, 100.0], None),   // text left column
            row(10, 0.6, [0.0, 0.0, 100.0, 20.0], None),    // doc_title
            row(4, 0.45, [0.0, 110.0, 100.0, 120.0], None), // abstract below 0.5 threshold
        ];
        let opts = LayoutOptions::for_model(LayoutModelType::DocLayoutPlusL);
        let regions = postprocess_layout(
            rows,
            None,
            LayoutModelType::DocLayoutPlusL,
            &opts,
            100.0,
            130.0,
        );
        let xs: Vec<_> = regions.iter().map(|r| (r.label, r.bbox.x1)).collect();
        assert_eq!(xs, vec![("doc_title", 0.0), ("text", 0.0), ("text", 55.0)]);
    }

    #[test]
    fn test_postprocess_clips_and_drops_degenerate() {
        let rows = vec![
            row(22, 0.9, [-5.0, -5.0, 50.5, 40.2], Some(0.0)),
            row(22, 0.9, [120.0, 0.0, 130.0, 10.0], Some(1.0)),
        ];
        let opts =
            LayoutOptions::for_model(LayoutModelType::DocLayoutV3).with_filter_overlaps(false);
        let regions = postprocess_layout(
            rows,
            None,
            LayoutModelType::DocLayoutV3,
            &opts,
            100.0,
            100.0,
        );
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].bbox, BoundingBox::new(0.0, 0.0, 50.0, 40.0));
    }

    fn region(label: &'static str, b: [f32; 4]) -> LayoutRegion {
        LayoutRegion {
            class_id: 0,
            label,
            score: 0.9,
            bbox: BoundingBox::new(b[0], b[1], b[2], b[3]),
            polygon: None,
            order: None,
        }
    }

    #[test]
    fn test_filter_overlap_regions() {
        let regions = vec![
            region("reference", [0.0, 0.0, 50.0, 50.0]),
            region("text", [0.0, 0.0, 100.0, 100.0]),
            region("text", [5.0, 5.0, 95.0, 95.0]), // duplicate of a larger text box
            region("inline_formula", [10.0, 10.0, 20.0, 20.0]), // inside text -> dropped
            region("image", [200.0, 0.0, 300.0, 100.0]),
            region("chart", [205.0, 5.0, 295.0, 95.0]), // vision vs vision -> both kept
            region("text", [400.0, 0.0, 403.0, 100.0]), // too thin
        ];
        let kept: Vec<_> = filter_overlap_regions(regions)
            .iter()
            .map(|r| r.label)
            .collect();
        assert_eq!(kept, vec!["text", "image", "chart"]);
    }

    #[test]
    fn test_polygon_overlap_small() {
        let square = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let inner = vec![[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]];
        let triangle = vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        assert!((polygon_overlap_small(&square, &inner) - 1.0).abs() < 0.02);
        assert!((polygon_overlap_small(&square, &triangle) - 1.0).abs() < 0.02);
        let far = vec![[20.0, 20.0], [30.0, 20.0], [30.0, 30.0]];
        assert_eq!(polygon_overlap_small(&square, &far), 0.0);
        // Two opposite triangles of the same square barely overlap.
        let other = vec![[10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(polygon_overlap_small(&triangle, &other) < 0.05);
    }

    #[test]
    fn test_region_polygon_from_mask() {
        // 8x8 mask over a 80x80 image; foreground is an L shape.
        let mut data = vec![0i32; 64];
        for y in 0..8 {
            for x in 0..8 {
                if x < 3 || y >= 5 {
                    data[y * 8 + x] = 1;
                }
            }
        }
        let masks = DetrMasks {
            data,
            height: 8,
            width: 8,
        };
        let poly = region_polygon(
            &masks,
            0,
            &BoundingBox::new(0.0, 0.0, 80.0, 80.0),
            80.0,
            80.0,
        )
        .unwrap();
        assert!(poly.len() >= 6, "L shape should keep its notch: {:?}", poly);
        // A fully filled mask is just the rectangle.
        let full = DetrMasks {
            data: vec![1; 64],
            height: 8,
            width: 8,
        };
        assert!(region_polygon(
            &full,
            0,
            &BoundingBox::new(0.0, 0.0, 80.0, 80.0),
            80.0,
            80.0
        )
        .is_none());
    }
}
