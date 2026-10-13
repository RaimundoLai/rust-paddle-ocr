#include "mnn_wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <memory>

// C++11 compatible make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_ptr(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Global mutex to serialize MNN inference calls
// MNN's internal thread pool has a limit of MNN_THREAD_POOL_MAX_TASKS (default=2)
// This mutex ensures only one inference runs at a time, avoiding thread pool exhaustion
static std::mutex g_mnn_inference_mutex;

// ============== Internal Structures ==============

struct MNN_SharedRuntime
{
    MNN::BackendConfig backend_config;
    MNN::ScheduleConfig schedule_config;
    int thread_count;
    int precision_mode;
};

struct MNN_InferenceEngine
{
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::Session *default_session;
    std::mutex mutex;
    std::string last_error;

    std::vector<int> input_shape;
    std::vector<int> output_shape;
    MNN::Tensor *input_tensor;
    MNN::Tensor *output_tensor;

    MNN_SharedRuntime *runtime; // Optional shared runtime
    bool owns_runtime;

    MNN_InferenceEngine() : default_session(nullptr), input_tensor(nullptr),
                            output_tensor(nullptr), runtime(nullptr), owns_runtime(false) {}
};

struct MNN_SingleSession
{
    MNN::Session *session;
    MNN_InferenceEngine *engine;
    std::string last_error;
    MNN::Tensor *input_tensor;
    MNN::Tensor *output_tensor;

    MNN_SingleSession() : session(nullptr), engine(nullptr),
                          input_tensor(nullptr), output_tensor(nullptr) {}
};

struct MNN_SessionPool
{
    MNN_InferenceEngine *engine;
    std::vector<MNN::Session *> sessions;
    std::vector<MNN::Tensor *> input_tensors;
    std::vector<MNN::Tensor *> output_tensors;

    std::mutex mutex;
    std::condition_variable cv;
    std::queue<size_t> available_sessions;
    std::string last_error;
};

// ============== Helper Functions ==============

// Initialize schedule and backend configs from MNNR_Config.
// Caller must ensure `schedule` and `backend` outlive any use of schedule.backendConfig.
static void init_schedule_config(MNN::ScheduleConfig &schedule, MNN::BackendConfig &backend, const MNNR_Config *config)
{
    schedule.type = (config) ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
    schedule.numThread = config ? config->thread_count : 4;
    if (schedule.numThread <= 0)
    {
        schedule.numThread = 4;
    }

    if (config)
    {
        switch (config->precision_mode)
        {
        case 1:
            backend.precision = MNN::BackendConfig::Precision_Low;
            break;
        case 2:
            backend.precision = MNN::BackendConfig::Precision_High;
            break;
        default:
            backend.precision = MNN::BackendConfig::Precision_Normal;
            break;
        }
    }
    schedule.backendConfig = &backend;
}

static bool init_engine_tensors(MNN_InferenceEngine *engine)
{
    if (!engine->interpreter || !engine->default_session)
    {
        return false;
    }

    // Get input tensor
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found";
        return false;
    }

    engine->input_tensor = input_map.begin()->second;
    auto input_shape_vec = engine->input_tensor->shape();
    engine->input_shape.assign(input_shape_vec.begin(), input_shape_vec.end());

    // Get output tensor
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return false;
    }

    engine->output_tensor = output_map.begin()->second;
    auto output_shape_vec = engine->output_tensor->shape();
    engine->output_shape.assign(output_shape_vec.begin(), output_shape_vec.end());

    return true;
}

// ============== Version & Info ==============

const char *mnnr_get_version(void)
{
    return MNN_VERSION;
}

// ============== Shared Runtime API ==============

MNN_SharedRuntime *mnnr_create_runtime(const MNNR_Config *config)
{
    auto runtime = new MNN_SharedRuntime();

    runtime->thread_count = config ? config->thread_count : 4;
    if (runtime->thread_count <= 0)
    {
        runtime->thread_count = 4;
    }

    runtime->precision_mode = config ? config->precision_mode : 0;

    runtime->schedule_config.type = (config) ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
    runtime->schedule_config.numThread = runtime->thread_count;

    switch (runtime->precision_mode)
    {
    case 1:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_Low;
        break;
    case 2:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_High;
        break;
    default:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_Normal;
        break;
    }
    runtime->schedule_config.backendConfig = &runtime->backend_config;

    return runtime;
}

void mnnr_destroy_runtime(MNN_SharedRuntime *runtime)
{
    delete runtime;
}

// ============== Inference Engine API ==============

MNN_InferenceEngine *mnnr_create_engine(
    const void *buffer,
    size_t size,
    const MNNR_Config *config)
{
    if (!buffer || size == 0)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create default session
    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    engine->default_session = engine->interpreter->createSession(schedule);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create default session";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

MNN_InferenceEngine *mnnr_create_engine_with_runtime(
    const void *buffer,
    size_t size,
    MNN_SharedRuntime *runtime)
{
    if (!buffer || size == 0 || !runtime)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();
    engine->runtime = runtime;
    engine->owns_runtime = false;

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create session using shared runtime config
    engine->default_session = engine->interpreter->createSession(runtime->schedule_config);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create session with shared runtime";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

void mnnr_destroy_engine(MNN_InferenceEngine *engine)
{
    if (engine)
    {
        if (engine->default_session && engine->interpreter)
        {
            engine->interpreter->releaseSession(engine->default_session);
        }
        delete engine;
    }
}

MNNR_ErrorCode mnnr_get_input_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    *out_ndims = engine->input_shape.size();
    for (size_t i = 0; i < engine->input_shape.size() && i < 8; i++)
    {
        dims[i] = static_cast<size_t>(engine->input_shape[i]);
    }

    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_get_output_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    *out_ndims = engine->output_shape.size();
    for (size_t i = 0; i < engine->output_shape.size() && i < 8; i++)
    {
        dims[i] = static_cast<size_t>(engine->output_shape[i]);
    }

    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_run_inference(
    MNN_InferenceEngine *engine,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!engine || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Use global lock to serialize MNN inference (thread pool limit)
    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Calculate expected sizes
    size_t expected_input = 1;
    for (int dim : engine->input_shape)
    {
        expected_input *= dim;
    }

    size_t expected_output = 1;
    for (int dim : engine->output_shape)
    {
        expected_output *= dim;
    }

    if (input_size != expected_input || output_size != expected_output)
    {
        engine->last_error = "Input/output size mismatch";
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Create host tensor and copy input data
    auto input_host = make_unique_ptr<MNN::Tensor>(engine->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));
    engine->input_tensor->copyFromHostTensor(input_host.get());

    // Run inference
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Copy output data
    auto output_host = make_unique_ptr<MNN::Tensor>(engine->output_tensor, MNN::Tensor::CAFFE);
    engine->output_tensor->copyToHostTensor(output_host.get());
    std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));

    return MNNR_SUCCESS;
}

const char *mnnr_get_last_error(const MNN_InferenceEngine *engine)
{
    if (!engine)
    {
        return "Engine is null";
    }
    return engine->last_error.c_str();
}

// ============== Session Pool API ==============

MNN_SessionPool *mnnr_create_session_pool(
    MNN_InferenceEngine *engine,
    size_t pool_size,
    const MNNR_Config *config)
{
    if (!engine || pool_size == 0)
    {
        return nullptr;
    }

    auto pool = new MNN_SessionPool();
    pool->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);

    // Create sessions
    for (size_t i = 0; i < pool_size; i++)
    {
        MNN::Session *session = engine->interpreter->createSession(schedule);
        if (!session)
        {
            // Cleanup on failure
            for (auto s : pool->sessions)
            {
                engine->interpreter->releaseSession(s);
            }
            // Note: input/output tensors are owned by MNN sessions, not by us.
            // They will be freed when sessions are released above.
            delete pool;
            return nullptr;
        }

        pool->sessions.push_back(session);
        pool->available_sessions.push(i);

        // Get input/output tensors for this session
        auto input_map = engine->interpreter->getSessionInputAll(session);
        auto output_map = engine->interpreter->getSessionOutputAll(session);

        pool->input_tensors.push_back(input_map.begin()->second);
        pool->output_tensors.push_back(output_map.begin()->second);
    }

    return pool;
}

void mnnr_destroy_session_pool(MNN_SessionPool *pool)
{
    if (pool)
    {
        for (auto session : pool->sessions)
        {
            if (pool->engine && pool->engine->interpreter)
            {
                pool->engine->interpreter->releaseSession(session);
            }
        }
        delete pool;
    }
}

MNNR_ErrorCode mnnr_session_pool_run(
    MNN_SessionPool *pool,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!pool || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Acquire a session (this will block if all sessions are busy)
    size_t session_idx;
    {
        std::unique_lock<std::mutex> lock(pool->mutex);
        pool->cv.wait(lock, [pool]
                      { return !pool->available_sessions.empty(); });
        session_idx = pool->available_sessions.front();
        pool->available_sessions.pop();
    }

    // Run inference with global lock to serialize MNN thread pool access
    MNNR_ErrorCode result = MNNR_SUCCESS;

    auto *session = pool->sessions[session_idx];
    auto *input_tensor = pool->input_tensors[session_idx];
    auto *output_tensor = pool->output_tensors[session_idx];

    // Create host tensor and copy input (can be done outside the global lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = pool->engine->interpreter->runSession(session);
        if (code != MNN::NO_ERROR)
        {
            pool->last_error = "Session pool inference failed";
            result = MNNR_ERROR_RUNTIME_ERROR;
        }
        else
        {
            // Copy output
            auto output_host = make_unique_ptr<MNN::Tensor>(output_tensor, MNN::Tensor::CAFFE);
            output_tensor->copyToHostTensor(output_host.get());
            std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
        }
    }

    // Release session
    {
        std::lock_guard<std::mutex> lock(pool->mutex);
        pool->available_sessions.push(session_idx);
    }
    pool->cv.notify_one();

    return result;
}

size_t mnnr_session_pool_available(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return 0;
    }
    std::lock_guard<std::mutex> lock(const_cast<MNN_SessionPool *>(pool)->mutex);
    return pool->available_sessions.size();
}

const char *mnnr_session_pool_get_last_error(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return "Pool is null";
    }
    return pool->last_error.c_str();
}

// ============== Single Session API ==============

MNN_SingleSession *mnnr_create_session(
    MNN_InferenceEngine *engine,
    const MNNR_Config *config)
{
    if (!engine)
    {
        return nullptr;
    }

    auto session = new MNN_SingleSession();
    session->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    session->session = engine->interpreter->createSession(schedule);

    if (!session->session)
    {
        delete session;
        return nullptr;
    }

    // Get tensors
    auto input_map = engine->interpreter->getSessionInputAll(session->session);
    auto output_map = engine->interpreter->getSessionOutputAll(session->session);

    if (input_map.empty() || output_map.empty())
    {
        engine->interpreter->releaseSession(session->session);
        delete session;
        return nullptr;
    }

    session->input_tensor = input_map.begin()->second;
    session->output_tensor = output_map.begin()->second;

    return session;
}

void mnnr_destroy_session(MNN_SingleSession *session)
{
    if (session)
    {
        if (session->session && session->engine && session->engine->interpreter)
        {
            session->engine->interpreter->releaseSession(session->session);
        }
        delete session;
    }
}

MNNR_ErrorCode mnnr_run_inference_with_session(
    MNN_SingleSession *session,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!session || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Create host tensor and copy input (outside lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(session->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        session->input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = session->engine->interpreter->runSession(session->session);
        if (code != MNN::NO_ERROR)
        {
            session->last_error = "Session inference failed";
            return MNNR_ERROR_RUNTIME_ERROR;
        }

        // Copy output
        auto output_host = make_unique_ptr<MNN::Tensor>(session->output_tensor, MNN::Tensor::CAFFE);
        session->output_tensor->copyToHostTensor(output_host.get());
        std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
    }

    return MNNR_SUCCESS;
}

const char *mnnr_session_get_last_error(const MNN_SingleSession *session)
{
    if (!session)
    {
        return "Session is null";
    }
    return session->last_error.c_str();
}

// ============== Dynamic Shape API ==============

MNNR_ErrorCode mnnr_run_inference_dynamic(
    MNN_InferenceEngine *engine,
    const float *input_data,
    const size_t *input_dims,
    size_t input_ndims,
    float **output_data,
    size_t *output_size,
    size_t *output_dims,
    size_t *output_ndims)
{
    if (!engine || !input_data || !input_dims || !output_data || !output_size || !output_dims || !output_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Build new input shape
    std::vector<int> new_shape(input_ndims);
    size_t total_input_size = 1;
    for (size_t i = 0; i < input_ndims; i++)
    {
        new_shape[i] = static_cast<int>(input_dims[i]);
        total_input_size *= input_dims[i];
    }

    // Resize input tensor
    engine->interpreter->resizeTensor(engine->input_tensor, new_shape);
    engine->interpreter->resizeSession(engine->default_session);

    // Get the updated input tensor after resize
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found after resize";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->input_tensor = input_map.begin()->second;

    // Create host tensor and copy input data
    auto input_host = make_unique_ptr<MNN::Tensor>(engine->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, total_input_size * sizeof(float));
    engine->input_tensor->copyFromHostTensor(input_host.get());

    // Run inference
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Dynamic inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Get output tensor after inference
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->output_tensor = output_map.begin()->second;

    // Get output shape
    auto output_shape = engine->output_tensor->shape();
    *output_ndims = output_shape.size();
    size_t total_output_size = 1;
    for (size_t i = 0; i < output_shape.size() && i < 8; i++)
    {
        output_dims[i] = static_cast<size_t>(output_shape[i]);
        total_output_size *= output_shape[i];
    }
    *output_size = total_output_size;

    // Allocate output buffer
    *output_data = new float[total_output_size];

    // Copy output data
    auto output_host = make_unique_ptr<MNN::Tensor>(engine->output_tensor, MNN::Tensor::CAFFE);
    engine->output_tensor->copyToHostTensor(output_host.get());
    std::memcpy(*output_data, output_host->host<float>(), total_output_size * sizeof(float));

    return MNNR_SUCCESS;
}

void mnnr_free_output(float *output_data)
{
    delete[] output_data;
}

// ============== Multi-Tensor Module API ==============

namespace MNNExpr = MNN::Express;

struct MNN_Module
{
    std::shared_ptr<MNNExpr::Module> module;
    std::shared_ptr<MNNExpr::Executor::RuntimeManager> runtime;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string last_error;
};

static void write_error(char *buffer, size_t buffer_size, const std::string &message)
{
    if (buffer == nullptr || buffer_size == 0)
    {
        return;
    }
    size_t n = message.size() < buffer_size - 1 ? message.size() : buffer_size - 1;
    std::memcpy(buffer, message.data(), n);
    buffer[n] = '\0';
}

static std::vector<std::string> collect_names(const char *const *names, size_t count)
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

MNN_Module *mnnr_module_create(
    const void *buffer,
    size_t size,
    const char *const *input_names,
    size_t num_inputs,
    const char *const *output_names,
    size_t num_outputs,
    const MNNR_Config *config,
    char *error_buffer,
    size_t error_buffer_size)
{
    if (buffer == nullptr || size == 0)
    {
        write_error(error_buffer, error_buffer_size, "Model buffer is empty");
        return nullptr;
    }

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);

    std::shared_ptr<MNNExpr::Executor::RuntimeManager> runtime(
        MNNExpr::Executor::RuntimeManager::createRuntimeManager(schedule));
    if (!runtime)
    {
        write_error(error_buffer, error_buffer_size, "Failed to create runtime manager");
        return nullptr;
    }

    MNNExpr::Module::Config module_config;
    module_config.shapeMutable = true;

    std::vector<std::string> inputs = collect_names(input_names, num_inputs);
    std::vector<std::string> outputs = collect_names(output_names, num_outputs);

    std::shared_ptr<MNNExpr::Module> module;
    {
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
        module.reset(MNNExpr::Module::load(inputs, outputs, static_cast<const uint8_t *>(buffer), size,
                                           runtime, &module_config));
    }
    if (!module)
    {
        write_error(error_buffer, error_buffer_size, "Failed to load module from buffer");
        return nullptr;
    }

    const MNNExpr::Module::Info *info = module->getInfo();
    auto handle = make_unique_ptr<MNN_Module>();
    handle->module = module;
    handle->runtime = runtime;
    handle->input_names = (inputs.empty() && info != nullptr) ? info->inputNames : inputs;
    handle->output_names = (outputs.empty() && info != nullptr) ? info->outputNames : outputs;
    if (handle->input_names.empty() || handle->output_names.empty())
    {
        write_error(error_buffer, error_buffer_size, "Module has no inputs or outputs");
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
        handle->module.reset();
        return nullptr;
    }
    return handle.release();
}

void mnnr_module_destroy(MNN_Module *module)
{
    if (module == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    module->module.reset();
    module->runtime.reset();
    delete module;
}

size_t mnnr_module_input_count(const MNN_Module *module)
{
    return module ? module->input_names.size() : 0;
}

size_t mnnr_module_output_count(const MNN_Module *module)
{
    return module ? module->output_names.size() : 0;
}

const char *mnnr_module_input_name(const MNN_Module *module, size_t index)
{
    if (module == nullptr || index >= module->input_names.size())
    {
        return nullptr;
    }
    return module->input_names[index].c_str();
}

const char *mnnr_module_output_name(const MNN_Module *module, size_t index)
{
    if (module == nullptr || index >= module->output_names.size())
    {
        return nullptr;
    }
    return module->output_names[index].c_str();
}

MNNR_ErrorCode mnnr_module_forward(
    MNN_Module *module,
    const MNNR_InputTensor *inputs,
    size_t num_inputs,
    MNNR_OutputTensor *outputs,
    size_t num_outputs)
{
    if (module == nullptr || outputs == nullptr || (inputs == nullptr && num_inputs > 0))
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    if (num_inputs != module->input_names.size() || num_outputs < module->output_names.size())
    {
        module->last_error = "Input/output count mismatch";
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    for (size_t i = 0; i < num_outputs; i++)
    {
        outputs[i].data = nullptr;
        outputs[i].element_count = 0;
        outputs[i].ndims = 0;
    }

    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

    std::vector<MNNExpr::VARP> input_vars;
    input_vars.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++)
    {
        const MNNR_InputTensor &tensor = inputs[i];
        if (tensor.data == nullptr || tensor.dims == nullptr || tensor.ndims == 0 || tensor.ndims > 8)
        {
            module->last_error = "Invalid input tensor " + module->input_names[i];
            return MNNR_ERROR_INVALID_PARAMETER;
        }
        std::vector<int> dims(tensor.dims, tensor.dims + tensor.ndims);
        size_t count = 1;
        for (int d : dims)
        {
            if (d < 0)
            {
                module->last_error = "Negative dimension in input " + module->input_names[i];
                return MNNR_ERROR_INVALID_PARAMETER;
            }
            count *= static_cast<size_t>(d);
        }
        MNNExpr::VARP var = MNNExpr::_Input(dims, MNNExpr::NCHW, halide_type_of<float>());
        float *dst = var->writeMap<float>();
        if (dst == nullptr && count > 0)
        {
            module->last_error = "Failed to map input " + module->input_names[i];
            return MNNR_ERROR_RUNTIME_ERROR;
        }
        if (count > 0)
        {
            std::memcpy(dst, tensor.data, count * sizeof(float));
        }
        input_vars.push_back(var);
    }

    std::vector<MNNExpr::VARP> results = module->module->onForward(input_vars);
    if (results.size() < module->output_names.size())
    {
        module->last_error = "Module forward failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    for (size_t i = 0; i < module->output_names.size(); i++)
    {
        MNNExpr::VARP var = results[i];
        const MNNExpr::Variable::Info *info = var.get() ? var->getInfo() : nullptr;
        if (info == nullptr)
        {
            mnnr_module_free_outputs(outputs, num_outputs);
            module->last_error = "Failed to compute output " + module->output_names[i];
            return MNNR_ERROR_RUNTIME_ERROR;
        }
        if (info->order == MNNExpr::NC4HW4)
        {
            var = MNNExpr::_Convert(var, MNNExpr::NCHW);
            info = var->getInfo();
        }
        bool is_float = info != nullptr && info->type.code == halide_type_float && info->type.bits == 32;
        if (info != nullptr && !is_float && !(info->type.code == halide_type_int && info->type.bits == 32))
        {
            var = MNNExpr::_Cast(var, halide_type_of<int32_t>());
            info = var->getInfo();
        }
        if (info == nullptr || info->dim.size() > 8)
        {
            mnnr_module_free_outputs(outputs, num_outputs);
            module->last_error = "Unsupported output " + module->output_names[i];
            return MNNR_ERROR_UNSUPPORTED;
        }

        MNNR_OutputTensor &out = outputs[i];
        out.dtype = is_float ? MNNR_DTYPE_FLOAT32 : MNNR_DTYPE_INT32;
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
            mnnr_module_free_outputs(outputs, num_outputs);
            return MNNR_ERROR_OUT_OF_MEMORY;
        }
        const void *src = is_float ? static_cast<const void *>(var->readMap<float>())
                                   : static_cast<const void *>(var->readMap<int32_t>());
        if (src == nullptr && out.element_count > 0)
        {
            mnnr_module_free_outputs(outputs, num_outputs);
            module->last_error = "Failed to read output " + module->output_names[i];
            return MNNR_ERROR_RUNTIME_ERROR;
        }
        if (out.element_count > 0)
        {
            std::memcpy(out.data, src, bytes);
        }
    }
    return MNNR_SUCCESS;
}

void mnnr_module_free_outputs(MNNR_OutputTensor *outputs, size_t num_outputs)
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

const char *mnnr_module_get_last_error(const MNN_Module *module)
{
    return module ? module->last_error.c_str() : "Invalid module";
}
