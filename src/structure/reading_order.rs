//! Reading order for layout regions without a model-predicted order.
//!
//! Uses a recursive XY-cut: split the region set by horizontal whitespace bands first,
//! then by vertical gutters (columns), and recurse. Boxes that cannot be separated are
//! ordered top-to-bottom, then left-to-right.

use super::geometry::BoundingBox;

/// Return the indices of `boxes` in reading order
pub fn xycut_order(boxes: &[BoundingBox]) -> Vec<usize> {
    let indices: Vec<usize> = (0..boxes.len()).collect();
    let mut order = Vec::with_capacity(boxes.len());
    cut(boxes, indices, true, &mut order, 0);
    order
}

const MAX_DEPTH: usize = 64;

fn cut(
    boxes: &[BoundingBox],
    indices: Vec<usize>,
    horizontal_first: bool,
    out: &mut Vec<usize>,
    depth: usize,
) {
    if indices.len() <= 1 || depth > MAX_DEPTH {
        let mut rest = indices;
        sort_top_left(boxes, &mut rest);
        out.extend(rest);
        return;
    }
    let axes = if horizontal_first {
        [true, false]
    } else {
        [false, true]
    };
    for &by_y in &axes {
        let groups = split_by_gaps(boxes, &indices, by_y);
        if groups.len() > 1 {
            for g in groups {
                // After splitting along one axis, try the other axis first inside each part.
                cut(boxes, g, !by_y, out, depth + 1);
            }
            return;
        }
    }
    let mut rest = indices;
    sort_top_left(boxes, &mut rest);
    out.extend(rest);
}

fn sort_top_left(boxes: &[BoundingBox], indices: &mut [usize]) {
    indices.sort_by(|&a, &b| {
        let (ba, bb) = (&boxes[a], &boxes[b]);
        ba.y1
            .partial_cmp(&bb.y1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                ba.x1
                    .partial_cmp(&bb.x1)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
}

/// Group boxes whose projections onto one axis overlap; groups are ordered along the axis
fn split_by_gaps(boxes: &[BoundingBox], indices: &[usize], by_y: bool) -> Vec<Vec<usize>> {
    let span = |i: usize| -> (f32, f32) {
        let b = &boxes[i];
        if by_y {
            (b.y1, b.y2)
        } else {
            (b.x1, b.x2)
        }
    };
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        span(a)
            .0
            .partial_cmp(&span(b).0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_end = f32::NEG_INFINITY;
    for i in sorted {
        let (start, end) = span(i);
        match groups.last_mut() {
            // Touching or overlapping spans stay in the same group.
            Some(g) if start < current_end => {
                g.push(i);
                current_end = current_end.max(end);
            }
            _ => {
                groups.push(vec![i]);
                current_end = end;
            }
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x1: f32, y1: f32, x2: f32, y2: f32) -> BoundingBox {
        BoundingBox::new(x1, y1, x2, y2)
    }

    #[test]
    fn test_single_column_top_to_bottom() {
        let boxes = vec![
            b(0.0, 50.0, 100.0, 60.0),
            b(0.0, 0.0, 100.0, 10.0),
            b(0.0, 20.0, 100.0, 30.0),
        ];
        assert_eq!(xycut_order(&boxes), vec![1, 2, 0]);
    }

    #[test]
    fn test_two_columns_with_full_width_title() {
        let boxes = vec![
            b(0.0, 30.0, 45.0, 100.0),   // left column body
            b(55.0, 30.0, 100.0, 60.0),  // right column top
            b(0.0, 0.0, 100.0, 20.0),    // title spanning both columns
            b(55.0, 65.0, 100.0, 100.0), // right column bottom
        ];
        assert_eq!(xycut_order(&boxes), vec![2, 0, 1, 3]);
    }

    #[test]
    fn test_footer_after_columns() {
        let boxes = vec![
            b(0.0, 120.0, 100.0, 130.0), // footer
            b(55.0, 0.0, 100.0, 100.0),  // right column
            b(0.0, 0.0, 45.0, 100.0),    // left column
        ];
        assert_eq!(xycut_order(&boxes), vec![2, 1, 0]);
    }

    #[test]
    fn test_overlapping_boxes_fall_back_to_position() {
        let boxes = vec![b(10.0, 5.0, 60.0, 50.0), b(0.0, 0.0, 50.0, 40.0)];
        assert_eq!(xycut_order(&boxes), vec![1, 0]);
    }

    #[test]
    fn test_empty_and_single() {
        assert!(xycut_order(&[]).is_empty());
        assert_eq!(xycut_order(&[b(0.0, 0.0, 1.0, 1.0)]), vec![0]);
    }
}
