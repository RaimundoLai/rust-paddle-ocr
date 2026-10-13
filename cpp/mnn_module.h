#ifndef OCR_RS_MNN_MODULE_H
#define OCR_RS_MNN_MODULE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Multi-tensor model API built on MNN::Express::Module.
    //
    // Complements mnn-runtime (single float session API) for models with several
    // named inputs/outputs, per-call dynamic shapes, and int32 outputs. It links
    // against the same MNN library that mnn-runtime-sys provides.

    typedef struct OcrRsMnnModule OcrRsMnnModule;

    typedef enum
    {
        OCR_RS_MNN_OK = 0,
        OCR_RS_MNN_INVALID_PARAMETER = 1,
        OCR_RS_MNN_OUT_OF_MEMORY = 2,
        OCR_RS_MNN_RUNTIME_ERROR = 3,
        OCR_RS_MNN_UNSUPPORTED = 4
    } OcrRsMnnStatus;

    typedef enum
    {
        OCR_RS_MNN_DTYPE_FLOAT32 = 0,
        OCR_RS_MNN_DTYPE_INT32 = 1
    } OcrRsMnnDataType;

    typedef struct
    {
        int32_t thread_count;   // <= 0 selects 4
        int32_t precision_mode; // 0 = normal, 1 = low, 2 = high
        int32_t forward_type;   // MNNForwardType
    } OcrRsMnnConfig;

    typedef struct
    {
        const float *data;
        const int32_t *dims;
        size_t ndims;
    } OcrRsMnnInput;

    // Output tensor owned by the bridge; release with ocr_rs_mnn_module_free_outputs
    typedef struct
    {
        int32_t dtype; // OcrRsMnnDataType
        void *data;
        size_t element_count;
        int32_t dims[8];
        size_t ndims;
    } OcrRsMnnOutput;

    // Create a module from model bytes. Empty name lists use the model's declared
    // inputs/outputs. error_buffer (optional) receives a message when NULL is returned.
    OcrRsMnnModule *ocr_rs_mnn_module_create(
        const void *buffer,
        size_t size,
        const char *const *input_names,
        size_t num_inputs,
        const char *const *output_names,
        size_t num_outputs,
        const OcrRsMnnConfig *config,
        char *error_buffer,
        size_t error_buffer_size);

    void ocr_rs_mnn_module_destroy(OcrRsMnnModule *module);

    size_t ocr_rs_mnn_module_input_count(const OcrRsMnnModule *module);
    size_t ocr_rs_mnn_module_output_count(const OcrRsMnnModule *module);
    const char *ocr_rs_mnn_module_input_name(const OcrRsMnnModule *module, size_t index);
    const char *ocr_rs_mnn_module_output_name(const OcrRsMnnModule *module, size_t index);

    // Run the module (serialized internally). inputs follow the module input order and
    // outputs must have room for ocr_rs_mnn_module_output_count() entries.
    int32_t ocr_rs_mnn_module_forward(
        OcrRsMnnModule *module,
        const OcrRsMnnInput *inputs,
        size_t num_inputs,
        OcrRsMnnOutput *outputs,
        size_t num_outputs);

    void ocr_rs_mnn_module_free_outputs(OcrRsMnnOutput *outputs, size_t num_outputs);

    // Last error message of the module (valid until the next call on it)
    const char *ocr_rs_mnn_module_last_error(const OcrRsMnnModule *module);

#ifdef __cplusplus
}
#endif

#endif // OCR_RS_MNN_MODULE_H
