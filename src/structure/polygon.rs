//! Polygon extraction from binary instance masks.

use image::GrayImage;
use imageproc::contours::{find_contours, BorderType};

/// Signed-area magnitude of a closed polygon (shoelace formula)
pub fn polygon_area(points: &[[f32; 2]]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for i in 0..points.len() {
        let [x1, y1] = points[i];
        let [x2, y2] = points[(i + 1) % points.len()];
        sum += x1 * y2 - x2 * y1;
    }
    sum.abs() / 2.0
}

/// Perimeter of a closed polygon
pub fn polygon_perimeter(points: &[[f32; 2]]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    (0..points.len())
        .map(|i| distance(points[i], points[(i + 1) % points.len()]))
        .sum()
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn point_segment_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return distance(p, a);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0);
    distance(p, [a[0] + t * dx, a[1] + t * dy])
}

/// Douglas-Peucker simplification of an open polyline (keeps both endpoints)
fn simplify_open(points: &[[f32; 2]], epsilon: f32, out: &mut Vec<[f32; 2]>) {
    if points.len() < 3 {
        out.extend_from_slice(&points[..points.len().saturating_sub(1)]);
        return;
    }
    let (first, last) = (points[0], points[points.len() - 1]);
    let (mut max_d, mut max_i) = (0.0f32, 0usize);
    for (i, &p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let d = point_segment_distance(p, first, last);
        if d > max_d {
            max_d = d;
            max_i = i;
        }
    }
    if max_d > epsilon {
        simplify_open(&points[..=max_i], epsilon, out);
        simplify_open(&points[max_i..], epsilon, out);
    } else {
        out.push(first);
    }
}

/// Douglas-Peucker simplification of a closed contour
pub fn simplify_closed(points: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
    if points.len() <= 3 {
        return points.to_vec();
    }
    // Split the ring at the point farthest from the first point.
    let far = (1..points.len())
        .max_by(|&a, &b| {
            distance(points[0], points[a])
                .partial_cmp(&distance(points[0], points[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let mut ring: Vec<[f32; 2]> = points.to_vec();
    ring.push(points[0]);
    let mut out = Vec::new();
    simplify_open(&ring[..=far], epsilon, &mut out);
    simplify_open(&ring[far..], epsilon, &mut out);
    out
}

/// Outer contour of the largest foreground component, simplified.
///
/// `mask` holds non-zero foreground pixels. Returns points in mask pixel coordinates.
pub fn mask_to_polygon(mask: &GrayImage, epsilon_ratio: f32) -> Option<Vec<[f32; 2]>> {
    let contours = find_contours::<i32>(mask);
    let best = contours
        .iter()
        .filter(|c| c.border_type == BorderType::Outer && c.points.len() >= 3)
        .map(|c| {
            let pts: Vec<[f32; 2]> = c.points.iter().map(|p| [p.x as f32, p.y as f32]).collect();
            (polygon_area(&pts), pts)
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))?;
    let pts = best.1;
    let epsilon = (polygon_perimeter(&pts) * epsilon_ratio).max(1.0);
    let simplified = simplify_closed(&pts, epsilon);
    if simplified.len() >= 3 {
        Some(simplified)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_area_and_perimeter() {
        let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        assert_eq!(polygon_area(&square), 16.0);
        assert_eq!(polygon_perimeter(&square), 16.0);
        assert_eq!(polygon_area(&square[..2]), 0.0);
    }

    #[test]
    fn test_simplify_closed_removes_collinear_points() {
        let pts = [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [4.0, 4.0],
            [2.0, 4.0],
            [0.0, 4.0],
            [0.0, 2.0],
        ];
        let s = simplify_closed(&pts, 0.5);
        assert_eq!(s.len(), 4);
        assert!((polygon_area(&s) - 16.0).abs() < 1e-4);
    }

    #[test]
    fn test_mask_to_polygon_picks_largest_component() {
        let mut mask = GrayImage::new(20, 20);
        for y in 2..12 {
            for x in 2..15 {
                mask.put_pixel(x, y, Luma([1]));
            }
        }
        mask.put_pixel(18, 18, Luma([1]));
        let poly = mask_to_polygon(&mask, 0.004).unwrap();
        assert_eq!(poly.len(), 4);
        let b = super::super::geometry::BoundingBox::from_points(&poly).unwrap();
        assert_eq!((b.x1, b.y1, b.x2, b.y2), (2.0, 2.0, 14.0, 11.0));
    }

    #[test]
    fn test_mask_to_polygon_empty() {
        assert!(mask_to_polygon(&GrayImage::new(5, 5), 0.004).is_none());
    }
}
