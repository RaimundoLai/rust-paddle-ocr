#include "mnn_module.h"

#include <MNN/MNNForwardType.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace
{
    namespace Expr = MNN::Express;

    // Module forward calls share MNN's default executor; serialize them.
    std::mutex g_module_mutex;

    void write_error(char *buffer, size_t buffer_size, const std::string &message)
    {
        if (buffer == nullptr || buffer_size == 0)
        {
            return;
        }
        size_t n = message.size() < buffer_size - 1 ? message.size() : buffer_size - 1;
        std::memcpy(buffer, message.data(), n);
        buffer[n] = '\0';
    }

    std::vector<std::string> collect_names(const char *const *names, size_t count)
    {
        std::vector<std::string> result;
        for (size_t i = 0; i < count; i++)
        {
            if (names != nullptr && names[i] != nullptr)
            {
                result.emplace_back(names[i]);
            }
        }
        return result;
    }
} // namespace

struct OcrRsMnnModule
{
    std::shared_ptr<Expr::Module> module;
    std::shared_ptr<Expr::Executor::RuntimeManager> runtime;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string last_error;
};

extern "C" OcrRsMnnModule *ocr_rs_mnn_module_create(
    const void *buffer,
    size_t size,
    const char *const *input_names,
    size_t num_inputs,
    const char *const *output_names,
    size_t num_outputs,
    const OcrRsMnnConfig *config,
    char *error_buffer,
    size_t error_buffer_size)
{
    if (buffer == nullptr || size == 0)
    {
        write_error(error_buffer, error_buffer_size, "Model buffer is empty");
        return nullptr;
    }
    try
    {
        MNN::ScheduleConfig schedule;
        MNN::BackendConfig backend;
        schedule.type = config ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
        schedule.numThread = (config && config->thread_count > 0) ? config->thread_count : 4;
        int precision = config ? config->precision_mode : 0;
        backend.precision = precision == 1   ? MNN::BackendConfig::Precision_Low
                            : precision == 2 ? MNN::BackendConfig::Precision_High
                                             : MNN::BackendConfig::Precision_Normal;
        schedule.backendConfig = &backend;

        std::shared_ptr<Expr::Executor::RuntimeManager> runtime(
            Expr::Executor::RuntimeManager::createRuntimeManager(schedule));
        if (!runtime)
        {
            write_error(error_buffer, error_buffer_size, "Failed to create MNN runtime manager");
            return nullptr;
        }

        Expr::Module::Config module_config;
        module_config.shapeMutable = true;
        std::vector<std::string> inputs = collect_names(input_names, num_inputs);
        std::vector<std::string> outputs = collect_names(output_names, num_outputs);

        std::shared_ptr<Expr::Module> module;
        {
            std::lock_guard<std::mutex> lock(g_module_mutex);
            module.reset(Expr::Module::load(inputs, outputs, static_cast<const uint8_t *>(buffer), size,
                                            runtime, &module_config));
        }
        if (!module)
        {
            write_error(error_buffer, error_buffer_size, "Failed to load MNN module from buffer");
            return nullptr;
        }

        const Expr::Module::Info *info = module->getInfo();
        std::unique_ptr<OcrRsMnnModule> handle(new OcrRsMnnModule());
        handle->module = module;
        handle->runtime = runtime;
        handle->input_names = (inputs.empty() && info != nullptr) ? info->inputNames : inputs;
        handle->output_names = (outputs.empty() && info != nullptr) ? info->outputNames : outputs;
        if (handle->input_names.empty() || handle->output_names.empty())
        {
            write_error(error_buffer, error_buffer_size, "MNN module has no inputs or outputs");
            std::lock_guard<std::mutex> lock(g_module_mutex);
            handle->module.reset();
            return nullptr;
        }
        return handle.release();
    }
    catch (const std::exception &e)
    {
        write_error(error_buffer, error_buffer_size, e.what());
    }
    catch (...)
    {
        write_error(error_buffer, error_buffer_size, "Unknown exception while loading MNN module");
    }
    return nullptr;
}

extern "C" void ocr_rs_mnn_module_destroy(OcrRsMnnModule *module)
{
    if (module == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_module_mutex);
    module->module.reset();
    module->runtime.reset();
    delete module;
}

extern "C" size_t ocr_rs_mnn_module_input_count(const OcrRsMnnModule *module)
{
    return module ? module->input_names.size() : 0;
}

extern "C" size_t ocr_rs_mnn_module_output_count(const OcrRsMnnModule *module)
{
    return module ? module->output_names.size() : 0;
}

extern "C" const char *ocr_rs_mnn_module_input_name(const OcrRsMnnModule *module, size_t index)
{
    if (module == nullptr || index >= module->input_names.size())
    {
        return nullptr;
    }
    return module->input_names[index].c_str();
}

extern "C" const char *ocr_rs_mnn_module_output_name(const OcrRsMnnModule *module, size_t index)
{
    if (module == nullptr || index >= module->output_names.size())
    {
        return nullptr;
    }
    return module->output_names[index].c_str();
}

extern "C" void ocr_rs_mnn_module_free_outputs(OcrRsMnnOutput *outputs, size_t num_outputs)
{
    if (outputs == nullptr)
    {
        return;
    }
    for (size_t i = 0; i < num_outputs; i++)
    {
        std::free(outputs[i].data);
        outputs[i].data = nullptr;
        outputs[i].element_count = 0;
    }
}

extern "C" int32_t ocr_rs_mnn_module_forward(
    OcrRsMnnModule *module,
    const OcrRsMnnInput *inputs,
    size_t num_inputs,
    OcrRsMnnOutput *outputs,
    size_t num_outputs)
{
    if (module == nullptr || outputs == nullptr || (inputs == nullptr && num_inputs > 0))
    {
        return OCR_RS_MNN_INVALID_PARAMETER;
    }
    if (num_inputs != module->input_names.size() || num_outputs < module->output_names.size())
    {
        module->last_error = "Input/output count mismatch";
        return OCR_RS_MNN_INVALID_PARAMETER;
    }
    for (size_t i = 0; i < num_outputs; i++)
    {
        outputs[i].data = nullptr;
        outputs[i].element_count = 0;
        outputs[i].ndims = 0;
    }

    try
    {
        std::lock_guard<std::mutex> lock(g_module_mutex);

        std::vector<Expr::VARP> input_vars;
        input_vars.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; i++)
        {
            const OcrRsMnnInput &tensor = inputs[i];
            if (tensor.data == nullptr || tensor.dims == nullptr || tensor.ndims == 0 || tensor.ndims > 8)
            {
                module->last_error = "Invalid input tensor " + module->input_names[i];
                return OCR_RS_MNN_INVALID_PARAMETER;
            }
            std::vector<int> dims(tensor.dims, tensor.dims + tensor.ndims);
            size_t count = 1;
            for (int d : dims)
            {
                if (d < 0)
                {
                    module->last_error = "Negative dimension in input " + module->input_names[i];
                    return OCR_RS_MNN_INVALID_PARAMETER;
                }
                count *= static_cast<size_t>(d);
            }
            Expr::VARP var = Expr::_Input(dims, Expr::NCHW, halide_type_of<float>());
            float *dst = var->writeMap<float>();
            if (dst == nullptr && count > 0)
            {
                module->last_error = "Failed to map input " + module->input_names[i];
                return OCR_RS_MNN_RUNTIME_ERROR;
            }
            if (count > 0)
            {
                std::memcpy(dst, tensor.data, count * sizeof(float));
            }
            input_vars.push_back(var);
        }

        std::vector<Expr::VARP> results = module->module->onForward(input_vars);
        if (results.size() < module->output_names.size())
        {
            module->last_error = "MNN module forward failed";
            return OCR_RS_MNN_RUNTIME_ERROR;
        }

        for (size_t i = 0; i < module->output_names.size(); i++)
        {
            Expr::VARP var = results[i];
            const Expr::Variable::Info *info = var.get() ? var->getInfo() : nullptr;
            if (info == nullptr)
            {
                ocr_rs_mnn_module_free_outputs(outputs, num_outputs);
                module->last_error = "Failed to compute output " + module->output_names[i];
                return OCR_RS_MNN_RUNTIME_ERROR;
            }
            if (info->order == Expr::NC4HW4)
            {
                var = Expr::_Convert(var, Expr::NCHW);
                info = var->getInfo();
            }
            bool is_float = info != nullptr && info->type.code == halide_type_float && info->type.bits == 32;
            if (info != nullptr && !is_float && !(info->type.code == halide_type_int && info->type.bits == 32))
            {
                var = Expr::_Cast(var, halide_type_of<int32_t>());
                info = var->getInfo();
            }
            if (info == nullptr || info->dim.size() > 8)
            {
                ocr_rs_mnn_module_free_outputs(outputs, num_outputs);
                module->last_error = "Unsupported output " + module->output_names[i];
                return OCR_RS_MNN_UNSUPPORTED;
            }

            OcrRsMnnOutput &out = outputs[i];
            out.dtype = is_float ? OCR_RS_MNN_DTYPE_FLOAT32 : OCR_RS_MNN_DTYPE_INT32;
            out.ndims = info->dim.size();
            for (size_t d = 0; d < out.ndims; d++)
            {
                out.dims[d] = info->dim[d];
            }
            out.element_count = static_cast<size_t>(info->size);
            size_t bytes = out.element_count * 4;
            out.data = std::malloc(bytes > 0 ? bytes : 4);
            if (out.data == nullptr)
            {
                ocr_rs_mnn_module_free_outputs(outputs, num_outputs);
                return OCR_RS_MNN_OUT_OF_MEMORY;
            }
            const void *src = is_float ? static_cast<const void *>(var->readMap<float>())
                                       : static_cast<const void *>(var->readMap<int32_t>());
            if (src == nullptr && out.element_count > 0)
            {
                ocr_rs_mnn_module_free_outputs(outputs, num_outputs);
                module->last_error = "Failed to read output " + module->output_names[i];
                return OCR_RS_MNN_RUNTIME_ERROR;
            }
            if (out.element_count > 0)
            {
                std::memcpy(out.data, src, bytes);
            }
        }
        return OCR_RS_MNN_OK;
    }
    catch (const std::exception &e)
    {
        ocr_rs_mnn_module_free_outputs(outputs, num_outputs);
        module->last_error = e.what();
    }
    catch (...)
    {
        ocr_rs_mnn_module_free_outputs(outputs, num_outputs);
        module->last_error = "Unknown exception during MNN module forward";
    }
    return OCR_RS_MNN_RUNTIME_ERROR;
}

extern "C" const char *ocr_rs_mnn_module_last_error(const OcrRsMnnModule *module)
{
    return module ? module->last_error.c_str() : "Invalid module";
}
