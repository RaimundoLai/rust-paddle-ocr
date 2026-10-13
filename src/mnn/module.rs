//! Multi-tensor models on MNN's Express Module API.
//!
//! `mnn-runtime` covers single float tensors through MNN sessions. Structure models need
//! several named inputs and outputs, per-call dynamic shapes, and int32 outputs, which
//! this thin bridge (`cpp/mnn_module.cpp`, linked against the same MNN) provides.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use ndarray::{ArrayD, ArrayViewD, IxDyn};

use super::{Backend, InferenceConfig, MnnError, PrecisionMode, Result};

#[repr(C)]
struct OcrRsMnnModule {
    _private: [u8; 0],
}

#[repr(C)]
struct OcrRsMnnConfig {
    thread_count: i32,
    precision_mode: i32,
    forward_type: i32,
}

#[repr(C)]
struct OcrRsMnnInput {
    data: *const f32,
    dims: *const i32,
    ndims: usize,
}

#[repr(C)]
struct OcrRsMnnOutput {
    dtype: i32,
    data: *mut c_void,
    element_count: usize,
    dims: [i32; 8],
    ndims: usize,
}

const STATUS_OK: i32 = 0;
const DTYPE_INT32: i32 = 1;

extern "C" {
    fn ocr_rs_mnn_module_create(
        buffer: *const c_void,
        size: usize,
        input_names: *const *const c_char,
        num_inputs: usize,
        output_names: *const *const c_char,
        num_outputs: usize,
        config: *const OcrRsMnnConfig,
        error_buffer: *mut c_char,
        error_buffer_size: usize,
    ) -> *mut OcrRsMnnModule;
    fn ocr_rs_mnn_module_destroy(module: *mut OcrRsMnnModule);
    fn ocr_rs_mnn_module_input_count(module: *const OcrRsMnnModule) -> usize;
    fn ocr_rs_mnn_module_output_count(module: *const OcrRsMnnModule) -> usize;
    fn ocr_rs_mnn_module_input_name(module: *const OcrRsMnnModule, index: usize) -> *const c_char;
    fn ocr_rs_mnn_module_output_name(module: *const OcrRsMnnModule, index: usize) -> *const c_char;
    fn ocr_rs_mnn_module_forward(
        module: *mut OcrRsMnnModule,
        inputs: *const OcrRsMnnInput,
        num_inputs: usize,
        outputs: *mut OcrRsMnnOutput,
        num_outputs: usize,
    ) -> i32;
    fn ocr_rs_mnn_module_free_outputs(outputs: *mut OcrRsMnnOutput, num_outputs: usize);
    fn ocr_rs_mnn_module_last_error(module: *const OcrRsMnnModule) -> *const c_char;
}

/// MNNForwardType value for a backend
fn forward_type(backend: Backend) -> i32 {
    match backend {
        Backend::CPU => 0,
        Backend::Metal => 1,
        Backend::CUDA => 2,
        Backend::OpenCL => 3,
        Backend::CoreML => 5,
        Backend::OpenGL => 6,
        Backend::Vulkan => 7,
    }
}

fn to_native_config(config: &InferenceConfig) -> OcrRsMnnConfig {
    OcrRsMnnConfig {
        thread_count: config.thread_count,
        precision_mode: match config.precision_mode {
            PrecisionMode::Normal => 0,
            PrecisionMode::Low => 1,
            PrecisionMode::High => 2,
        },
        forward_type: forward_type(config.backend),
    }
}

/// Element data of a model output tensor
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    /// 32-bit float elements
    F32(Vec<f32>),
    /// 32-bit integer elements (integer and boolean outputs are widened to i32)
    I32(Vec<i32>),
}

/// Model output tensor with its shape
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tensor elements in row-major order
    pub data: TensorData,
}

impl Tensor {
    /// Number of elements
    pub fn len(&self) -> usize {
        match &self.data {
            TensorData::F32(v) => v.len(),
            TensorData::I32(v) => v.len(),
        }
    }

    /// Whether the tensor has no elements
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow float elements, if the tensor holds floats
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            TensorData::F32(v) => Some(v),
            TensorData::I32(_) => None,
        }
    }

    /// Borrow integer elements, if the tensor holds integers
    pub fn as_i32(&self) -> Option<&[i32]> {
        match &self.data {
            TensorData::I32(v) => Some(v),
            TensorData::F32(_) => None,
        }
    }

    /// Elements converted to f32
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match &self.data {
            TensorData::F32(v) => v.clone(),
            TensorData::I32(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Convert into an f32 ndarray (integer data is converted)
    pub fn into_f32_array(self) -> Result<ArrayD<f32>> {
        let shape = self.shape;
        let data = match self.data {
            TensorData::F32(v) => v,
            TensorData::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        };
        ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| MnnError::RuntimeError(format!("Invalid output shape: {}", e)))
    }
}

/// MNN model with several named inputs and outputs
///
/// Runs through MNN's Module API, so it supports dynamic input shapes on every call,
/// models with control flow, and integer outputs. Calls are serialized internally.
pub struct MultiModel {
    ptr: NonNull<OcrRsMnnModule>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl MultiModel {
    /// Create a model from bytes.
    ///
    /// `input_names` / `output_names` select and order the model tensors; pass empty
    /// slices to use the order declared by the model.
    pub fn from_buffer(
        model_buffer: &[u8],
        input_names: &[&str],
        output_names: &[&str],
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        if model_buffer.is_empty() {
            return Err(MnnError::InvalidParameter(
                "Model data is empty".to_string(),
            ));
        }
        let to_cstrings = |names: &[&str]| -> Result<Vec<CString>> {
            names
                .iter()
                .map(|n| {
                    CString::new(*n).map_err(|_| {
                        MnnError::InvalidParameter(format!("Invalid tensor name: {}", n))
                    })
                })
                .collect()
        };
        let inputs = to_cstrings(input_names)?;
        let outputs = to_cstrings(output_names)?;
        let input_ptrs: Vec<*const c_char> = inputs.iter().map(|s| s.as_ptr()).collect();
        let output_ptrs: Vec<*const c_char> = outputs.iter().map(|s| s.as_ptr()).collect();

        let native_config = to_native_config(&config.unwrap_or_default());
        let mut error_buffer = vec![0 as c_char; 512];
        let module_ptr = unsafe {
            ocr_rs_mnn_module_create(
                model_buffer.as_ptr() as *const c_void,
                model_buffer.len(),
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_ptrs.as_ptr(),
                output_ptrs.len(),
                &native_config,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        let ptr = NonNull::new(module_ptr).ok_or_else(|| {
            let msg = unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            MnnError::ModelLoadFailed(if msg.is_empty() {
                "Module creation failed".to_string()
            } else {
                msg
            })
        })?;

        let raw = ptr.as_ptr();
        let read = |p: *const c_char| {
            if p.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        };
        let input_names = (0..unsafe { ocr_rs_mnn_module_input_count(raw) })
            .map(|i| read(unsafe { ocr_rs_mnn_module_input_name(raw, i) }))
            .collect();
        let output_names = (0..unsafe { ocr_rs_mnn_module_output_count(raw) })
            .map(|i| read(unsafe { ocr_rs_mnn_module_output_name(raw, i) }))
            .collect();

        Ok(MultiModel {
            ptr,
            input_names,
            output_names,
        })
    }

    /// Create a model from a file
    pub fn from_file(
        model_path: impl AsRef<Path>,
        input_names: &[&str],
        output_names: &[&str],
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        let model_buffer = std::fs::read(model_path.as_ref())
            .map_err(|e| MnnError::ModelLoadFailed(format!("Failed to read model file: {}", e)))?;
        Self::from_buffer(&model_buffer, input_names, output_names, config)
    }

    /// Input names in call order
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Output names in result order
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Run inference with inputs in [`input_names`](Self::input_names) order
    pub fn run(&self, inputs: &[ArrayViewD<f32>]) -> Result<Vec<Tensor>> {
        if inputs.len() != self.input_names.len() {
            return Err(MnnError::InvalidParameter(format!(
                "Expected {} inputs, got {}",
                self.input_names.len(),
                inputs.len()
            )));
        }
        let contiguous: Vec<_> = inputs.iter().map(|a| a.as_standard_layout()).collect();
        let dims: Vec<Vec<i32>> = inputs
            .iter()
            .map(|a| a.shape().iter().map(|&d| d as i32).collect())
            .collect();
        let native_inputs: Vec<OcrRsMnnInput> = contiguous
            .iter()
            .zip(dims.iter())
            .map(|(a, d)| OcrRsMnnInput {
                data: a.as_ptr(),
                dims: d.as_ptr(),
                ndims: d.len(),
            })
            .collect();
        let mut native_outputs: Vec<OcrRsMnnOutput> = (0..self.output_names.len())
            .map(|_| OcrRsMnnOutput {
                dtype: 0,
                data: std::ptr::null_mut(),
                element_count: 0,
                dims: [0; 8],
                ndims: 0,
            })
            .collect();

        let status = unsafe {
            ocr_rs_mnn_module_forward(
                self.ptr.as_ptr(),
                native_inputs.as_ptr(),
                native_inputs.len(),
                native_outputs.as_mut_ptr(),
                native_outputs.len(),
            )
        };
        if status != STATUS_OK {
            let msg = unsafe {
                let c_str = ocr_rs_mnn_module_last_error(self.ptr.as_ptr());
                if c_str.is_null() {
                    "Unknown error".to_string()
                } else {
                    CStr::from_ptr(c_str).to_string_lossy().into_owned()
                }
            };
            return Err(MnnError::RuntimeError(msg));
        }

        let results = native_outputs
            .iter()
            .map(|o| {
                let shape: Vec<usize> = o.dims[..o.ndims.min(8)]
                    .iter()
                    .map(|&d| d as usize)
                    .collect();
                let data = if o.dtype == DTYPE_INT32 {
                    TensorData::I32(if o.data.is_null() || o.element_count == 0 {
                        Vec::new()
                    } else {
                        unsafe { std::slice::from_raw_parts(o.data as *const i32, o.element_count) }
                            .to_vec()
                    })
                } else {
                    TensorData::F32(if o.data.is_null() || o.element_count == 0 {
                        Vec::new()
                    } else {
                        unsafe { std::slice::from_raw_parts(o.data as *const f32, o.element_count) }
                            .to_vec()
                    })
                };
                Tensor { shape, data }
            })
            .collect();
        unsafe {
            ocr_rs_mnn_module_free_outputs(native_outputs.as_mut_ptr(), native_outputs.len())
        };
        Ok(results)
    }

    /// Run inference with inputs given by name (any order)
    pub fn run_named(&self, inputs: &[(&str, ArrayViewD<f32>)]) -> Result<Vec<Tensor>> {
        let ordered = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .iter()
                    .find(|(n, _)| *n == name.as_str())
                    .map(|(_, a)| a.view())
                    .ok_or_else(|| MnnError::InvalidParameter(format!("Missing input {}", name)))
            })
            .collect::<Result<Vec<_>>>()?;
        self.run(&ordered)
    }
}

impl Drop for MultiModel {
    fn drop(&mut self) {
        unsafe { ocr_rs_mnn_module_destroy(self.ptr.as_ptr()) };
    }
}

// The native bridge serializes every call on a module.
unsafe impl Send for MultiModel {}
unsafe impl Sync for MultiModel {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_conversions() {
        let t = Tensor {
            shape: vec![2],
            data: TensorData::I32(vec![1, 2]),
        };
        assert_eq!(t.len(), 2);
        assert_eq!(t.as_i32(), Some(&[1, 2][..]));
        assert!(t.as_f32().is_none());
        assert_eq!(t.to_f32_vec(), vec![1.0, 2.0]);
        assert_eq!(t.into_f32_array().unwrap().shape(), &[2]);
    }

    #[test]
    fn test_forward_type_mapping() {
        assert_eq!(forward_type(Backend::CPU), 0);
        assert_eq!(forward_type(Backend::OpenCL), 3);
        assert_eq!(forward_type(Backend::Vulkan), 7);
    }

    #[test]
    fn test_invalid_model_bytes_are_rejected() {
        assert!(MultiModel::from_buffer(&[], &[], &[], None).is_err());
        assert!(MultiModel::from_buffer(&[1, 2, 3], &[], &[], None).is_err());
    }
}
