//! Document structure analysis example
//!
//! Runs layout detection, OCR, table, formula and (optionally) unwarping on one page and
//! prints the blocks in reading order followed by Markdown.
//!
//! Convert the models first:
//!   python script/convert_doc_structure.py --output models/structure
//!
//! Usage:
//!   cargo run --release --example doc_structure -- <image> [options]
//!
//! Options:
//!   --models <dir>        structure models directory (default: models/structure)
//!   --layout <v3|plus-l>  layout model (default: v3)
//!   --det <path>          OCR detection model (default: models/PP-OCRv5_mobile_det.mnn)
//!   --rec <path>          OCR recognition model (default: models/PP-OCRv5_mobile_rec.mnn)
//!   --keys <path>         OCR charset (default: models/ppocr_keys_v5.txt)
//!   --no-ocr              skip text recognition
//!   --no-table            skip table recognition
//!   --no-formula          skip formula recognition
//!   --no-inline-formula   only recognize display formulas
//!   --table-cells         use table classification and cell detection
//!   --unwarp              unwarp the page with UVDoc first

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use ocr_rs::structure::{
    BlockContent, DocStructureEngine, DocUnwarpModel, FormulaModel, LayoutModel, LayoutModelType,
    TableCellDetector, TableClassifier, TableRecognizer, TableStructureModel,
};
use ocr_rs::{InferenceConfig, OcrEngine};

struct Args {
    image: PathBuf,
    models: PathBuf,
    layout: LayoutModelType,
    det: PathBuf,
    rec: PathBuf,
    keys: PathBuf,
    ocr: bool,
    table: bool,
    formula: bool,
    inline_formula: bool,
    table_cells: bool,
    unwarp: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut args = Args {
        image: PathBuf::new(),
        models: PathBuf::from("models/structure"),
        layout: LayoutModelType::DocLayoutV3,
        det: PathBuf::from("models/PP-OCRv5_mobile_det.mnn"),
        rec: PathBuf::from("models/PP-OCRv5_mobile_rec.mnn"),
        keys: PathBuf::from("models/ppocr_keys_v5.txt"),
        ocr: true,
        table: true,
        formula: true,
        inline_formula: true,
        table_cells: false,
        unwarp: false,
    };
    let mut image = None;
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{} needs a value", name));
        match a.as_str() {
            "--models" => args.models = value("--models")?.into(),
            "--layout" => {
                args.layout = match value("--layout")?.as_str() {
                    "v3" => LayoutModelType::DocLayoutV3,
                    "plus-l" => LayoutModelType::DocLayoutPlusL,
                    other => return Err(format!("unknown layout model {}", other)),
                }
            }
            "--det" => args.det = value("--det")?.into(),
            "--rec" => args.rec = value("--rec")?.into(),
            "--keys" => args.keys = value("--keys")?.into(),
            "--no-ocr" => args.ocr = false,
            "--no-table" => args.table = false,
            "--no-formula" => args.formula = false,
            "--no-inline-formula" => args.inline_formula = false,
            "--table-cells" => args.table_cells = true,
            "--unwarp" => args.unwarp = true,
            other if other.starts_with("--") => return Err(format!("unknown option {}", other)),
            other => image = Some(PathBuf::from(other)),
        }
    }
    args.image = image.ok_or("missing image path")?;
    Ok(args)
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = parse_args().map_err(|e| {
        eprintln!("{}\nUsage: doc_structure <image> [--models dir] [--layout v3|plus-l] [--no-ocr] [--no-table] [--no-formula] [--table-cells] [--unwarp]", e);
        e
    })?;
    let config = InferenceConfig::new().with_threads(4);
    let m = |name: &str| args.models.join(name);

    let start = Instant::now();
    let layout = match args.layout {
        LayoutModelType::DocLayoutV3 => {
            LayoutModel::from_file(m("PP-DocLayoutV3.mnn"), args.layout, Some(config.clone()))?
        }
        LayoutModelType::DocLayoutPlusL => LayoutModel::from_file(
            m("PP-DocLayout_plus-L.mnn"),
            args.layout,
            Some(config.clone()),
        )?,
    };
    let mut engine = DocStructureEngine::new(layout).with_inline_formulas(args.inline_formula);
    if args.ocr {
        engine = engine.with_ocr(OcrEngine::new(&args.det, &args.rec, &args.keys, None)?);
    }
    if args.table {
        let structure = TableStructureModel::from_files(
            m("SLANet_plus_encoder.mnn"),
            m("SLANet_plus_step.mnn"),
            Some(config.clone()),
        )?;
        let mut table = TableRecognizer::new(structure);
        if args.table_cells {
            table = table
                .with_classifier(TableClassifier::from_file(
                    m("PP-LCNet_x1_0_table_cls.mnn"),
                    Some(config.clone()),
                )?)
                .with_cell_detectors(
                    Some(TableCellDetector::from_file(
                        m("RT-DETR-L_wired_table_cell_det.mnn"),
                        Some(config.clone()),
                    )?),
                    Some(TableCellDetector::from_file(
                        m("RT-DETR-L_wireless_table_cell_det.mnn"),
                        Some(config.clone()),
                    )?),
                );
        }
        engine = engine.with_table(table);
    }
    if args.formula {
        engine = engine.with_formula(FormulaModel::from_files(
            m("PP-FormulaNet_plus-L_encoder.mnn"),
            m("PP-FormulaNet_plus-L_decoder.mnn"),
            m("PP-FormulaNet_plus-L_vocab.txt"),
            Some(config.clone()),
        )?);
    }
    if args.unwarp {
        engine = engine.with_unwarp(DocUnwarpModel::from_file(
            m("UVDoc.mnn"),
            Some(config.clone()),
        )?);
    }
    println!("Models loaded in {:.2?}", start.elapsed());

    let image = image::open(&args.image)?;
    let start = Instant::now();
    let result = engine.analyze(&image)?;
    println!(
        "Analyzed {}x{} page in {:.2?}: {} blocks, {} OCR lines\n",
        result.width,
        result.height,
        start.elapsed(),
        result.blocks.len(),
        result.lines.len()
    );

    for (i, block) in result.blocks.iter().enumerate() {
        let b = &block.bbox;
        let order = block
            .order
            .map(|o| o.to_string())
            .unwrap_or_else(|| "-".to_string());
        let summary = match &block.content {
            BlockContent::Text(t) => t.chars().take(60).collect::<String>(),
            BlockContent::Table(t) => format!(
                "table {} cells, {} chars of HTML",
                t.cells.len(),
                t.html.len()
            ),
            BlockContent::Formula { latex, inline } => {
                format!("{}{}", if *inline { "(inline) " } else { "" }, latex)
            }
            BlockContent::Image => "[image]".to_string(),
            BlockContent::Empty => String::new(),
        };
        println!(
            "{:>3} order {:>3} {:<18} {:.2} [{:.0},{:.0},{:.0},{:.0}]{} {}",
            i,
            order,
            block.label,
            block.score,
            b.x1,
            b.y1,
            b.x2,
            b.y2,
            if block.polygon.is_some() { " poly" } else { "" },
            summary
        );
    }
    println!("\n----- Markdown -----\n{}", result.to_markdown());
    Ok(())
}
