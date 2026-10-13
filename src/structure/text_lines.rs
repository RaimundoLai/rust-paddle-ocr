//! Page-level OCR lines and helpers to attach them to layout regions.

use super::geometry::BoundingBox;

/// A recognized text line in page coordinates
#[derive(Debug, Clone, PartialEq)]
pub struct TextLine {
    /// Axis-aligned line box
    pub bbox: BoundingBox,
    /// Recognized text
    pub text: String,
    /// Recognition confidence
    pub score: f32,
}

impl TextLine {
    /// Create a text line
    pub fn new(bbox: BoundingBox, text: impl Into<String>, score: f32) -> Self {
        Self {
            bbox,
            text: text.into(),
            score,
        }
    }
}

/// Whether two boxes overlap by more than 3 px on both axes (PaddleX `get_sub_regions_ocr_res`)
pub fn overlaps_region(line: &BoundingBox, region: &BoundingBox) -> bool {
    let w = line.x2.min(region.x2) - line.x1.max(region.x1);
    let h = line.y2.min(region.y2) - line.y1.max(region.y1);
    w > 3.0 && h > 3.0
}

/// Assign each line to the eligible region it overlaps most; returns line indices per region.
///
/// Lines that overlap no eligible region are left unassigned.
pub fn assign_lines(
    lines: &[TextLine],
    regions: &[BoundingBox],
    eligible: &[bool],
) -> Vec<Vec<usize>> {
    let mut assigned = vec![Vec::new(); regions.len()];
    for (li, line) in lines.iter().enumerate() {
        let best = regions
            .iter()
            .enumerate()
            .filter(|(ri, r)| {
                eligible.get(*ri).copied().unwrap_or(true) && overlaps_region(&line.bbox, r)
            })
            .max_by(|(_, a), (_, b)| {
                line.bbox
                    .intersection_area(a)
                    .partial_cmp(&line.bbox.intersection_area(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(ri, _)| ri);
        if let Some(ri) = best {
            assigned[ri].push(li);
        }
    }
    assigned
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF   // Hiragana, Katakana
        | 0x3400..=0x4DBF // CJK Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0xF900..=0xFAFF // CJK compatibility ideographs
        | 0x3000..=0x303F // CJK punctuation
        | 0xFF00..=0xFFEF) // Full-width forms
}

/// Separator between two text fragments: none across CJK boundaries, a space otherwise
fn separator(prev: &str, next: &str) -> &'static str {
    match (prev.chars().last(), next.chars().next()) {
        (Some(a), Some(b)) if is_cjk(a) || is_cjk(b) => "",
        (Some(a), _) if a.is_whitespace() => "",
        (None, _) | (_, None) => "",
        _ => " ",
    }
}

/// Group line indices into visual rows, ordered top-to-bottom then left-to-right
pub fn group_rows(lines: &[&TextLine]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lines.len()).collect();
    order.sort_by(|&a, &b| {
        let (ca, cb) = (lines[a].bbox.center().1, lines[b].bbox.center().1);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rows: Vec<(BoundingBox, Vec<usize>)> = Vec::new();
    for i in order {
        let b = lines[i].bbox;
        let joined = rows.last_mut().filter(|(span, _)| {
            let overlap = b.y2.min(span.y2) - b.y1.max(span.y1);
            let min_h = b.height().min(span.height()).max(1.0);
            overlap / min_h >= 0.5
        });
        match joined {
            Some((span, members)) => {
                *span = span.union(&b);
                members.push(i);
            }
            None => rows.push((b, vec![i])),
        }
    }
    rows.into_iter()
        .map(|(_, mut members)| {
            members.sort_by(|&a, &b| {
                lines[a]
                    .bbox
                    .x1
                    .partial_cmp(&lines[b].bbox.x1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            members
        })
        .collect()
}

/// Join lines into a paragraph in reading order
pub fn join_lines(lines: &[&TextLine]) -> String {
    let mut out = String::new();
    for row in group_rows(lines) {
        for i in row {
            let text = lines[i].text.trim();
            if text.is_empty() {
                continue;
            }
            out.push_str(separator(&out, text));
            out.push_str(text);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(x1: f32, y1: f32, x2: f32, y2: f32, text: &str) -> TextLine {
        TextLine::new(BoundingBox::new(x1, y1, x2, y2), text, 0.9)
    }

    #[test]
    fn test_overlaps_region_requires_more_than_three_pixels() {
        let region = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        assert!(overlaps_region(
            &BoundingBox::new(90.0, 10.0, 120.0, 20.0),
            &region
        ));
        assert!(!overlaps_region(
            &BoundingBox::new(97.0, 10.0, 120.0, 20.0),
            &region
        ));
    }

    #[test]
    fn test_assign_lines_picks_largest_overlap_and_respects_eligibility() {
        let lines = vec![
            line(0.0, 0.0, 50.0, 10.0, "a"),
            line(60.0, 0.0, 120.0, 10.0, "b"),
            line(500.0, 0.0, 510.0, 10.0, "c"),
        ];
        let regions = vec![
            BoundingBox::new(0.0, 0.0, 70.0, 20.0),
            BoundingBox::new(65.0, 0.0, 200.0, 20.0),
        ];
        assert_eq!(
            assign_lines(&lines, &regions, &[true, true]),
            vec![vec![0], vec![1]]
        );
        assert_eq!(
            assign_lines(&lines, &regions, &[true, false]),
            vec![vec![0, 1], vec![]]
        );
    }

    #[test]
    fn test_join_lines_orders_rows_and_words() {
        let a = line(60.0, 0.0, 100.0, 10.0, "world");
        let b = line(0.0, 1.0, 50.0, 11.0, "hello");
        let c = line(0.0, 20.0, 50.0, 30.0, "next");
        assert_eq!(join_lines(&[&a, &b, &c]), "hello world next");
    }

    #[test]
    fn test_join_lines_cjk_has_no_spaces() {
        let a = line(0.0, 0.0, 50.0, 10.0, "你好");
        let b = line(0.0, 20.0, 50.0, 30.0, "世界");
        let c = line(0.0, 40.0, 50.0, 50.0, "OCR");
        assert_eq!(join_lines(&[&a, &b, &c]), "你好世界OCR");
    }

    #[test]
    fn test_join_lines_skips_empty() {
        let a = line(0.0, 0.0, 50.0, 10.0, "  ");
        let b = line(0.0, 20.0, 50.0, 30.0, "text");
        assert_eq!(join_lines(&[&a, &b]), "text");
        assert_eq!(join_lines(&[]), "");
    }
}
