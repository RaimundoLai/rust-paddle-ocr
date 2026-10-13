//! Compatibility checks for the historical MNN link-mode features.
//!
//! Native compilation and linking are owned by `mnn-runtime-sys`. This script
//! preserves the validation users of `ocr-rs` already rely on and compiles the
//! multi-tensor Module bridge against the MNN headers that `mnn-runtime-sys` exports.

use std::{env, path::Path};

mod build_support;

fn main() {
    for variable in ["MNN_INCLUDE_DIR", "MNN_LIB_DIR", "DOCS_RS"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    if env::var_os("DOCS_RS").is_some() || env::var_os("CARGO_FEATURE_DOCSRS").is_some() {
        return;
    }

    let dynamic = env::var_os("CARGO_FEATURE_MNN_DYNAMIC").is_some();
    let static_link = env::var_os("CARGO_FEATURE_MNN_STATIC").is_some();
    build_support::validate_link_features(dynamic, static_link)
        .unwrap_or_else(|error| panic!("{error}"));

    if build_support::requires_external_installation(dynamic, static_link) {
        validate_external_directory("MNN_INCLUDE_DIR");
        validate_external_directory("MNN_LIB_DIR");
    }

    build_module_bridge(dynamic);
}

/// MNN include directory exported by `mnn-runtime-sys`.
///
/// It prints `cargo:metadata=include=<dir>`, which reaches dependents as
/// `DEP_MNN_METADATA=include=<dir>` (or `DEP_MNN_INCLUDE` with the `cargo::metadata` form).
fn mnn_include_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = env::var_os("DEP_MNN_INCLUDE") {
        return Some(dir.into());
    }
    env::var("DEP_MNN_METADATA")
        .ok()
        .and_then(|value| value.strip_prefix("include=").map(std::path::PathBuf::from))
}

/// Compile `cpp/mnn_module.cpp` with the same flags `mnn-runtime-sys` uses for its bridge.
fn build_module_bridge(dynamic: bool) {
    println!("cargo:rerun-if-changed=cpp/mnn_module.cpp");
    println!("cargo:rerun-if-changed=cpp/mnn_module.h");
    let include = mnn_include_dir()
        .expect("mnn-runtime-sys did not export its MNN include directory (links = \"mnn\")");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .cpp_link_stdlib(None::<&str>)
        .include(&include)
        .include("cpp")
        .file("cpp/mnn_module.cpp")
        .warnings(true);
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build
            .flag_if_supported("/std:c++17")
            .flag_if_supported("/EHsc");
        let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .split(',')
            .any(|feature| feature == "crt-static");
        build.static_crt(crt_static);
        if dynamic {
            build.define("USING_MNN_DLL", None);
        }
    } else {
        build
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-fvisibility=hidden");
    }
    build.compile("ocr_rs_mnn_module");
}

fn validate_external_directory(variable: &str) {
    let value = env::var_os(variable).unwrap_or_else(|| {
        panic!("{variable} is required when using `mnn-dynamic` or `mnn-static`")
    });
    let path = Path::new(&value);
    assert!(
        path.is_dir(),
        "{variable}='{}' does not exist or is not a directory",
        path.display()
    );
}
