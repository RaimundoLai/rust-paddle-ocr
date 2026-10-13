//! Document structure tests against real converted models.
//!
//! Skipped unless both variables are set:
//! - `OCR_RS_STRUCTURE_MODELS`: output directory of `script/convert_doc_structure.py`
//! - `OCR_RS_STRUCTURE_IMAGES`: directory with PaddleX demo images `layout.jpg`,
//!   `table_recognition.jpg`, `general_formula_rec_001.png` and `doc_test.jpg`
//!   (https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/<name>)

use std::path::PathBuf;

use ndarray::ArrayD;
use ocr_rs::structure::{
    DocUnwarpModel, FormulaModel, LayoutModel, LayoutModelType, TableClassifier,
    TableStructureModel, TableType,
};
use ocr_rs::{InferenceConfig, MultiModel};

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let models = std::env::var_os("OCR_RS_STRUCTURE_MODELS")?;
    let images = std::env::var_os("OCR_RS_STRUCTURE_IMAGES")?;
    Some((PathBuf::from(models), PathBuf::from(images)))
}

macro_rules! require_dirs {
    () => {
        match dirs() {
            Some(d) => d,
            None => {
                eprintln!("skipped: set OCR_RS_STRUCTURE_MODELS and OCR_RS_STRUCTURE_IMAGES");
                return;
            }
        }
    };
}

fn config() -> Option<InferenceConfig> {
    Some(InferenceConfig::new().with_threads(4))
}

#[test]
fn layout_v3_detects_regions_in_reading_order() {
    let (models, images) = require_dirs!();
    let model = LayoutModel::from_file(
        models.join("PP-DocLayoutV3.mnn"),
        LayoutModelType::DocLayoutV3,
        config(),
    )
    .unwrap();
    let image = image::open(images.join("layout.jpg")).unwrap();
    let regions = model.detect(&image).unwrap();

    assert!(regions.len() >= 10, "too few regions: {}", regions.len());
    assert_eq!(regions.iter().filter(|r| r.label == "table").count(), 2);
    assert!(regions.iter().any(|r| r.label == "paragraph_title"));
    let (w, h) = (image.width() as f32, image.height() as f32);
    for r in &regions {
        assert!(
            r.bbox.is_valid()
                && r.bbox.x1 >= 0.0
                && r.bbox.y1 >= 0.0
                && r.bbox.x2 <= w
                && r.bbox.y2 <= h
        );
    }
    let orders: Vec<usize> = regions.iter().filter_map(|r| r.order).collect();
    assert_eq!(orders, (1..=orders.len()).collect::<Vec<_>>());
    // The left column ("Learning Curve") is read before the right column's text.
    let pos = |text_x: f32, label: &str| {
        regions
            .iter()
            .position(|r| r.label == label && r.bbox.x1 < text_x)
    };
    assert!(
        pos(200.0, "paragraph_title")
            < regions
                .iter()
                .position(|r| r.label == "text" && r.bbox.x1 > 300.0)
    );
}

#[test]
fn layout_plus_l_detects_regions() {
    let (models, images) = require_dirs!();
    let model = LayoutModel::from_file(
        models.join("PP-DocLayout_plus-L.mnn"),
        LayoutModelType::DocLayoutPlusL,
        config(),
    )
    .unwrap();
    let image = image::open(images.join("layout.jpg")).unwrap();
    let regions = model.detect(&image).unwrap();
    assert!(regions.len() >= 10, "too few regions: {}", regions.len());
    assert_eq!(regions.iter().filter(|r| r.label == "table").count(), 2);
    assert!(regions.iter().all(|r| r.polygon.is_none()));
}

#[test]
fn table_structure_matches_reference() {
    let (models, images) = require_dirs!();
    let model = TableStructureModel::from_files(
        models.join("SLANet_plus_encoder.mnn"),
        models.join("SLANet_plus_step.mnn"),
        config(),
    )
    .unwrap();
    let image = image::open(images.join("table_recognition.jpg")).unwrap();
    let s = model.predict(&image).unwrap();
    assert_eq!(s.cell_count(), 13);
    assert_eq!(s.cells.len(), 13);
    assert!(s
        .tokens
        .concat()
        .starts_with("<tr><td colspan=\"4\"></td></tr><tr><td></td>"));
    assert!(s.score > 0.9);
    for b in s.cell_boxes() {
        assert!(b.x2 <= image.width() as f32 + 1.0 && b.y2 <= image.height() as f32 + 1.0);
    }
}

#[test]
fn table_classifier_runs() {
    let (models, images) = require_dirs!();
    let model =
        TableClassifier::from_file(models.join("PP-LCNet_x1_0_table_cls.mnn"), config()).unwrap();
    let image = image::open(images.join("table_recognition.jpg")).unwrap();
    let (kind, score) = model.classify(&image).unwrap();
    assert!(matches!(kind, TableType::Wired | TableType::Wireless));
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn formula_recognition_matches_reference() {
    let (models, images) = require_dirs!();
    let model = FormulaModel::from_files(
        models.join("PP-FormulaNet_plus-L_encoder.mnn"),
        models.join("PP-FormulaNet_plus-L_decoder.mnn"),
        models.join("PP-FormulaNet_plus-L_vocab.txt"),
        config(),
    )
    .unwrap();
    let image = image::open(images.join("general_formula_rec_001.png")).unwrap();
    let result = model.recognize(&image).unwrap();
    assert!(
        result
            .latex
            .starts_with("\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}"),
        "unexpected LaTeX: {}",
        result.latex
    );
    assert!(result.token_count > 100);
}

#[test]
fn unwarp_keeps_input_size() {
    let (models, images) = require_dirs!();
    let model = DocUnwarpModel::from_file(models.join("UVDoc.mnn"), config()).unwrap();
    let image = image::open(images.join("doc_test.jpg")).unwrap();
    let out = model.unwarp(&image).unwrap();
    assert_eq!((out.width(), out.height()), (image.width(), image.height()));
}

#[test]
fn multi_model_rejects_wrong_input_count() {
    let (models, _) = require_dirs!();
    let model = MultiModel::from_file(
        models.join("PP-LCNet_x1_0_table_cls.mnn"),
        &["x"],
        &["fetch_name_0"],
        config(),
    )
    .unwrap();
    assert_eq!(model.input_names(), &["x".to_string()]);
    let a = ArrayD::<f32>::zeros(vec![1, 3, 224, 224]);
    assert!(model.run(&[a.view(), a.view()]).is_err());
    let out = model.run(&[a.view()]).unwrap();
    assert_eq!(out[0].shape, vec![1, 2]);
    assert!(MultiModel::from_buffer(&[1, 2, 3], &[], &[], None).is_err());
}

#[test]
fn multi_model_uses_declared_names_when_none_given() {
    let (models, _) = require_dirs!();
    let model = MultiModel::from_file(
        models.join("RT-DETR-L_wired_table_cell_det.mnn"),
        &[],
        &[],
        config(),
    )
    .unwrap();
    let mut inputs = model.input_names().to_vec();
    inputs.sort();
    assert_eq!(inputs, vec!["im_shape", "image", "scale_factor"]);
    assert_eq!(model.output_names().len(), 2);
    // Named inputs may be given in any order.
    let im_shape = ArrayD::<f32>::from_shape_vec(vec![1, 2], vec![640.0, 640.0]).unwrap();
    let image = ArrayD::<f32>::zeros(vec![1, 3, 640, 640]);
    let scale = ArrayD::<f32>::from_shape_vec(vec![1, 2], vec![1.0, 1.0]).unwrap();
    let out = model
        .run_named(&[
            ("scale_factor", scale.view()),
            ("image", image.view()),
            ("im_shape", im_shape.view()),
        ])
        .unwrap();
    assert_eq!(out.len(), 2);
    assert!(model.run_named(&[("image", image.view())]).is_err());
}
