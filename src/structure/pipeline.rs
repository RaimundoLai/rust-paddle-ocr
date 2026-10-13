//! End-to-end document structure pipeline.
//!
//! Orientation and unwarping (optional) → layout → formula recognition → page OCR with
//! formulas masked out → text assignment per region → table recognition → blocks in
//! reading order, exportable as Markdown.

use image::{DynamicImage, Rgb, RgbImage};

use super::formula::{FormulaModel, FormulaResult};
use super::geometry::BoundingBox;
use super::image_ops::crop_rgb;
use super::layout::{LayoutModel, LayoutRegion};
use super::table::{TableRecognizer, TableResult};
use super::text_lines::{assign_lines, join_lines, TextLine};
use super::unwarp::DocUnwarpModel;
use crate::engine::{rotate_by_angle, OcrEngine};
use crate::error::OcrResult;
use crate::ori::OriModel;

/// Role of a layout label in the pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Document title
    DocTitle,
    /// Section title
    ParagraphTitle,
    /// Text recognized by OCR
    Text,
    /// Table
    Table,
    /// Formula (display or inline)
    Formula,
    /// Picture-like region without text recognition
    Image,
}

/// Map a layout label to its pipeline role
pub fn region_kind(label: &str) -> RegionKind {
    match label {
        "doc_title" => RegionKind::DocTitle,
        "paragraph_title" => RegionKind::ParagraphTitle,
        "table" => RegionKind::Table,
        "formula" | "display_formula" | "inline_formula" => RegionKind::Formula,
        "image" | "chart" | "seal" | "header_image" | "footer_image" => RegionKind::Image,
        _ => RegionKind::Text,
    }
}

/// Labels left out of the Markdown export (page furniture)
const MARKDOWN_SKIP_LABELS: [&str; 7] = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
];

/// Recognized content of a block
#[derive(Debug, Clone, PartialEq)]
pub enum BlockContent {
    /// Text content (titles and paragraphs)
    Text(String),
    /// Table with HTML
    Table(TableResult),
    /// Formula; `inline` is set when it was embedded into a surrounding text block
    Formula {
        /// Recognized LaTeX
        latex: String,
        /// Embedded into another block's text as `$...$`
        inline: bool,
    },
    /// Image-like region
    Image,
    /// No content recognized (model not configured or nothing found)
    Empty,
}

/// One block of the analyzed document
#[derive(Debug, Clone, PartialEq)]
pub struct DocBlock {
    /// Layout label
    pub label: &'static str,
    /// Region box in page coordinates
    pub bbox: BoundingBox,
    /// Region polygon, when available
    pub polygon: Option<Vec<[f32; 2]>>,
    /// Layout score
    pub score: f32,
    /// 1-based reading order for main-flow blocks
    pub order: Option<usize>,
    /// Recognized content
    pub content: BlockContent,
}

/// Document analysis result
#[derive(Debug, Clone)]
pub struct DocStructureResult {
    /// Page width after preprocessing
    pub width: u32,
    /// Page height after preprocessing
    pub height: u32,
    /// Clockwise rotation (degrees) detected and undone before analysis
    pub rotation: i32,
    /// Whether the page was unwarped
    pub unwarped: bool,
    /// Page image the coordinates refer to, when it differs from the input
    pub preprocessed_image: Option<DynamicImage>,
    /// Blocks in reading order
    pub blocks: Vec<DocBlock>,
    /// Page-level OCR lines
    pub lines: Vec<TextLine>,
}

fn strip_html_wrappers(html: &str) -> String {
    html.replace("<html>", "")
        .replace("</html>", "")
        .replace("<body>", "")
        .replace("</body>", "")
}

impl DocStructureResult {
    /// Render the document as Markdown
    pub fn to_markdown(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for block in &self.blocks {
            if MARKDOWN_SKIP_LABELS.contains(&block.label) {
                continue;
            }
            let part = match (&block.content, region_kind(block.label)) {
                (BlockContent::Text(t), _) if t.trim().is_empty() => continue,
                (BlockContent::Text(t), RegionKind::DocTitle) => format!("# {}", t.trim()),
                (BlockContent::Text(t), RegionKind::ParagraphTitle) => format!("## {}", t.trim()),
                (BlockContent::Text(t), _) => t.trim().to_string(),
                (BlockContent::Table(table), _) => strip_html_wrappers(&table.html),
                (
                    BlockContent::Formula {
                        latex,
                        inline: false,
                    },
                    _,
                ) if !latex.is_empty() => {
                    format!("$$\n{}\n$$", latex)
                }
                _ => continue,
            };
            parts.push(part);
        }
        parts.join("\n\n")
    }

    /// Plain text of all text, formula and table blocks in reading order
    pub fn plain_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match &b.content {
                BlockContent::Text(t) if !t.trim().is_empty() => Some(t.trim().to_string()),
                BlockContent::Formula {
                    latex,
                    inline: false,
                } if !latex.is_empty() => Some(latex.clone()),
                BlockContent::Table(t) => Some(strip_html_wrappers(&t.html)),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Document structure pipeline
pub struct DocStructureEngine {
    layout: LayoutModel,
    ocr: Option<OcrEngine>,
    table: Option<TableRecognizer>,
    formula: Option<FormulaModel>,
    unwarp: Option<DocUnwarpModel>,
    orientation: Option<OriModel>,
    orientation_min_confidence: f32,
    recognize_inline_formulas: bool,
}

impl DocStructureEngine {
    /// Create a pipeline from a layout model; other stages are optional
    pub fn new(layout: LayoutModel) -> Self {
        Self {
            layout,
            ocr: None,
            table: None,
            formula: None,
            unwarp: None,
            orientation: None,
            orientation_min_confidence: 0.3,
            recognize_inline_formulas: true,
        }
    }

    /// OCR engine for text regions (should not include its own orientation model)
    pub fn with_ocr(mut self, ocr: OcrEngine) -> Self {
        self.ocr = Some(ocr);
        self
    }

    /// Table recognizer
    pub fn with_table(mut self, table: TableRecognizer) -> Self {
        self.table = Some(table);
        self
    }

    /// Formula recognizer
    pub fn with_formula(mut self, formula: FormulaModel) -> Self {
        self.formula = Some(formula);
        self
    }

    /// Document unwarping model
    pub fn with_unwarp(mut self, unwarp: DocUnwarpModel) -> Self {
        self.unwarp = Some(unwarp);
        self
    }

    /// Document orientation classifier (PP-LCNet_x1_0_doc_ori)
    pub fn with_orientation(mut self, orientation: OriModel, min_confidence: f32) -> Self {
        self.orientation = Some(orientation);
        self.orientation_min_confidence = min_confidence;
        self
    }

    /// Recognize inline formulas too (each costs one formula model run; default true)
    pub fn with_inline_formulas(mut self, enable: bool) -> Self {
        self.recognize_inline_formulas = enable;
        self
    }

    /// Layout model
    pub fn layout_model(&self) -> &LayoutModel {
        &self.layout
    }

    /// Analyze a document page
    pub fn analyze(&self, image: &DynamicImage) -> OcrResult<DocStructureResult> {
        // 1. Page preprocessing.
        let mut page = image.clone();
        let mut rotation = 0;
        if let Some(ori) = &self.orientation {
            let result = ori.classify(&page)?;
            if result.is_valid(self.orientation_min_confidence) && result.angle.rem_euclid(360) != 0
            {
                rotation = result.angle.rem_euclid(360);
                page = rotate_by_angle(&page, rotation);
            }
        }
        let unwarped = self.unwarp.is_some();
        if let Some(unwarp) = &self.unwarp {
            page = unwarp.unwarp(&page)?;
        }
        let modified = rotation != 0 || unwarped;
        let rgb = page.to_rgb8();
        let (width, height) = rgb.dimensions();

        // 2. Layout.
        let regions = self.layout.detect(&page)?;
        let kinds: Vec<RegionKind> = regions.iter().map(|r| region_kind(r.label)).collect();

        // 3. Formulas.
        let mut formulas: Vec<Option<FormulaResult>> = vec![None; regions.len()];
        if let Some(model) = &self.formula {
            for (i, region) in regions.iter().enumerate() {
                if kinds[i] != RegionKind::Formula {
                    continue;
                }
                if region.label == "inline_formula" && !self.recognize_inline_formulas {
                    continue;
                }
                if let Some((x, y, w, h)) = region.bbox.to_pixel_rect(width, height) {
                    let crop = DynamicImage::ImageRgb8(crop_rgb(&rgb, x, y, w, h));
                    formulas[i] = Some(model.recognize(&crop)?);
                }
            }
        }

        // 4. Page OCR with recognized formulas painted white.
        let mut lines: Vec<TextLine> = Vec::new();
        if let Some(ocr) = &self.ocr {
            let mut ocr_page = rgb.clone();
            for (i, region) in regions.iter().enumerate() {
                if formulas[i].is_some() {
                    paint_white(&mut ocr_page, &region.bbox);
                }
            }
            lines = run_ocr(ocr, &DynamicImage::ImageRgb8(ocr_page), 0.0, 0.0)?;
        }

        // 5. Assign OCR lines and inline formulas to text regions.
        let boxes: Vec<BoundingBox> = regions.iter().map(|r| r.bbox).collect();
        let text_like: Vec<bool> = kinds
            .iter()
            .map(|k| {
                matches!(
                    k,
                    RegionKind::Text | RegionKind::DocTitle | RegionKind::ParagraphTitle
                )
            })
            .collect();
        let mut text_lines = lines.clone();
        let first_formula_line = text_lines.len();
        let mut formula_line_owner: Vec<usize> = Vec::new();
        for (i, f) in formulas.iter().enumerate() {
            if let Some(f) = f {
                if !f.latex.is_empty() {
                    text_lines.push(TextLine::new(
                        regions[i].bbox,
                        format!("${}$", f.latex),
                        1.0,
                    ));
                    formula_line_owner.push(i);
                }
            }
        }
        let assigned = assign_lines(&text_lines, &boxes, &text_like);
        let mut inline_formula = vec![false; regions.len()];
        for members in &assigned {
            for &li in members {
                if li >= first_formula_line {
                    inline_formula[formula_line_owner[li - first_formula_line]] = true;
                }
            }
        }

        // 6. Build blocks.
        let mut blocks = Vec::with_capacity(regions.len());
        for (i, region) in regions.iter().enumerate() {
            let content = match kinds[i] {
                RegionKind::Text | RegionKind::DocTitle | RegionKind::ParagraphTitle => {
                    let members: Vec<&TextLine> =
                        assigned[i].iter().map(|&li| &text_lines[li]).collect();
                    if !members.is_empty() {
                        BlockContent::Text(join_lines(&members))
                    } else if let Some(ocr) = &self.ocr {
                        // Nothing matched at page level: recognize the region on its own.
                        match region.bbox.to_pixel_rect(width, height) {
                            Some((x, y, w, h)) => {
                                let crop = DynamicImage::ImageRgb8(crop_rgb(&rgb, x, y, w, h));
                                let crop_lines = run_ocr(ocr, &crop, x as f32, y as f32)?;
                                let refs: Vec<&TextLine> = crop_lines.iter().collect();
                                BlockContent::Text(join_lines(&refs))
                            }
                            None => BlockContent::Empty,
                        }
                    } else {
                        BlockContent::Empty
                    }
                }
                RegionKind::Table => match &self.table {
                    Some(table) => {
                        BlockContent::Table(table.recognize(&rgb, &region.bbox, &text_lines)?)
                    }
                    None => BlockContent::Empty,
                },
                RegionKind::Formula => match &formulas[i] {
                    Some(f) => BlockContent::Formula {
                        latex: f.latex.clone(),
                        inline: inline_formula[i],
                    },
                    None => BlockContent::Empty,
                },
                RegionKind::Image => BlockContent::Image,
            };
            blocks.push(block_from_region(region, content));
        }

        Ok(DocStructureResult {
            width,
            height,
            rotation,
            unwarped,
            preprocessed_image: if modified { Some(page) } else { None },
            blocks,
            lines,
        })
    }
}

fn block_from_region(region: &LayoutRegion, content: BlockContent) -> DocBlock {
    DocBlock {
        label: region.label,
        bbox: region.bbox,
        polygon: region.polygon.clone(),
        score: region.score,
        order: region.order,
        content,
    }
}

fn paint_white(image: &mut RgbImage, bbox: &BoundingBox) {
    if let Some((x, y, w, h)) = bbox.to_pixel_rect(image.width(), image.height()) {
        for yy in y..y + h {
            for xx in x..x + w {
                image.put_pixel(xx, yy, Rgb([255, 255, 255]));
            }
        }
    }
}

fn run_ocr(ocr: &OcrEngine, image: &DynamicImage, dx: f32, dy: f32) -> OcrResult<Vec<TextLine>> {
    Ok(ocr
        .recognize(image)?
        .into_iter()
        .map(|r| {
            let rect = r.bbox.rect;
            let bbox = BoundingBox::new(
                rect.left() as f32 + dx,
                rect.top() as f32 + dy,
                (rect.left() + rect.width() as i32) as f32 + dx,
                (rect.top() + rect.height() as i32) as f32 + dy,
            );
            TextLine::new(bbox, r.text, r.confidence)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::table::TableType;

    fn block(label: &'static str, content: BlockContent) -> DocBlock {
        DocBlock {
            label,
            bbox: BoundingBox::new(0.0, 0.0, 10.0, 10.0),
            polygon: None,
            score: 0.9,
            order: None,
            content,
        }
    }

    #[test]
    fn test_region_kind() {
        assert_eq!(region_kind("doc_title"), RegionKind::DocTitle);
        assert_eq!(region_kind("display_formula"), RegionKind::Formula);
        assert_eq!(region_kind("formula_number"), RegionKind::Text);
        assert_eq!(region_kind("chart"), RegionKind::Image);
        assert_eq!(region_kind("vertical_text"), RegionKind::Text);
    }

    #[test]
    fn test_to_markdown() {
        let result = DocStructureResult {
            width: 10,
            height: 10,
            rotation: 0,
            unwarped: false,
            preprocessed_image: None,
            blocks: vec![
                block("header", BlockContent::Text("Journal".into())),
                block("doc_title", BlockContent::Text("Title".into())),
                block("paragraph_title", BlockContent::Text("1 Intro".into())),
                block("text", BlockContent::Text("Body with $x$.".into())),
                block(
                    "inline_formula",
                    BlockContent::Formula {
                        latex: "x".into(),
                        inline: true,
                    },
                ),
                block(
                    "display_formula",
                    BlockContent::Formula {
                        latex: "E=mc^2".into(),
                        inline: false,
                    },
                ),
                block(
                    "table",
                    BlockContent::Table(TableResult {
                        html: "<html><body><table><tr><td>a</td></tr></table></body></html>".into(),
                        cells: vec![],
                        score: 1.0,
                        table_type: Some(TableType::Wired),
                    }),
                ),
                block("image", BlockContent::Image),
                block("text", BlockContent::Text("  ".into())),
            ],
            lines: vec![],
        };
        assert_eq!(
            result.to_markdown(),
            "# Title\n\n## 1 Intro\n\nBody with $x$.\n\n$$\nE=mc^2\n$$\n\n<table><tr><td>a</td></tr></table>"
        );
        assert_eq!(
            result.plain_text(),
            "Journal\nTitle\n1 Intro\nBody with $x$.\nE=mc^2\n<table><tr><td>a</td></tr></table>"
        );
    }

    #[test]
    fn test_paint_white() {
        let mut img = RgbImage::from_pixel(4, 4, Rgb([0, 0, 0]));
        paint_white(&mut img, &BoundingBox::new(1.0, 1.0, 3.0, 2.0));
        assert_eq!(img.get_pixel(1, 1).0, [255, 255, 255]);
        assert_eq!(img.get_pixel(2, 1).0, [255, 255, 255]);
        assert_eq!(img.get_pixel(1, 2).0, [0, 0, 0]);
    }
}
