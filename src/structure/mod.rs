//! Document structure analysis (PP-Structure style) on MNN.
//!
//! - [`pipeline`]: end-to-end [`DocStructureEngine`] producing ordered blocks and Markdown
//! - [`layout`]: layout detection with PP-DocLayoutV3 / PP-DocLayout_plus-L
//! - [`reading_order`]: XY-cut reading order for models without predicted order
//! - [`table`]: table structure (SLANet_plus), classification, cell detection and HTML
//! - [`formula`]: formula recognition (PP-FormulaNet_plus) to LaTeX
//! - [`text_lines`]: attaching page OCR lines to regions
//! - [`unwarp`]: document unwarping (UVDoc)
//! - [`geometry`]: box helpers shared by the structure modules

mod detr;
pub mod formula;
pub mod geometry;
pub mod image_ops;
pub mod layout;
pub mod pipeline;
pub mod polygon;
pub mod reading_order;
pub mod table;
pub mod text_lines;
pub mod unwarp;

pub use formula::{FormulaModel, FormulaResult, LatexVocab};
pub use geometry::BoundingBox;
pub use layout::{LayoutModel, LayoutModelType, LayoutOptions, LayoutRegion};
pub use pipeline::{BlockContent, DocBlock, DocStructureEngine, DocStructureResult, RegionKind};
pub use table::{
    TableCellDetector, TableClassifier, TableRecognizer, TableResult, TableStructure,
    TableStructureModel, TableType,
};
pub use text_lines::TextLine;
pub use unwarp::DocUnwarpModel;
