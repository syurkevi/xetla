/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "tests/utils/utils.hpp"
#include "xetla.hpp"

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include <chrono>
#include <map>
#include <tuple>
#include <utility>
#include <iostream>

bool done_warmup;
std::vector<sycl::event> allevents;

using keyMNK = std::tuple<size_t, size_t, size_t>;
using time_flops_bw = std::tuple<float, float, float>;


enum class kslicing_impl_t : uint8_t { none = 0, global = 1, local = 2 };

template<typename T>
T* fill_device_ptr(sycl::queue &queue, sycl::context &context, sycl::device &device, size_t elems) {
    T* dev_ptr = alloc_device_and_init<T>(
            elems,
            [](T *data, size_t idx) {
                data[idx] = static_cast<T>(random_float());
            },
            queue, device, context);
    return dev_ptr;
}

template <kslicing_impl_t kslicing_type = kslicing_impl_t::none>
void gemm_universal_run(uint32_t iter, size_t matrix_m=4096, size_t matrix_n=4096, size_t matrix_k=4096) {
    // Tips, the example demonstrates programming kernel with XeTLA, it works as expected with current configurations.
    // Please make sure you fully understand these configurations before you do any modifications, incomplete changes may lead to unexpected behaviors.
    // Please contact us for support.

    //GEMM_UNIVERSAL input size
    //size_t matrix_m = 4;
    //size_t matrix_n = 4096;
    //size_t matrix_k = 4096;

    size_t size_a = matrix_m * matrix_k;
    size_t size_b = matrix_k * matrix_n;
    size_t size_c = matrix_m * matrix_n;

    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
    using data_type_acc = float;

    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};

    //Define SYCL queue, context and device
    auto queue = sycl::queue(properties);
    //auto queue = sycl::queue();
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto A = alloc_device_and_init<data_type_a>(
            size_a,
            [](data_type_a *data, size_t idx) {
                data[idx] = static_cast<data_type_a>(random_float());
            },
            queue, device, context);
    auto B = alloc_device_and_init<data_type_b>(
            size_b,
            [](data_type_b *data, size_t idx) {
                data[idx] = static_cast<data_type_b>(random_float());
            },
            queue, device, context);
    auto C = alloc_device_and_init<data_type_c>(
            size_c,
            [](data_type_c *data, size_t idx) {
                data[idx] = static_cast<data_type_c>(0.0f);
            },
            queue, device, context);

    //Define the shape of workgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m
            = (kslicing_type != kslicing_impl_t::local) ? 256 : 64;
    constexpr uint32_t wg_tile_n
            = (kslicing_type != kslicing_impl_t::local) ? 256 : 128;

    // specify the range k_w/k_s by setting the corresponding ratio
    // splitk using global memory
    constexpr uint32_t num_global_splitk
            = (kslicing_type == kslicing_impl_t::global) ? 2 : 1;
    // splitk using local memory
    constexpr uint32_t num_local_splitk
            = (kslicing_type == kslicing_impl_t::local) ? 2 : 1;

    // Mirco-kernel configuration
    using tune_option = dict_t<
            elem_v_t<tune_key::param_optimizer_type,
                    tune_key_value::param_optimizer_decision_tree>,
            elem_t_t<tune_key::data_type_acc, data_type_acc>,
            elem_v_t<tune_key::dispatch_policy,
                    tune_key_value::dispatch_policy_kslicing>,
            elem_v_t<tune_key::global_kslicing_ratio, num_global_splitk>,
            elem_v_t<tune_key::local_kslicing_ratio, num_local_splitk>,
            elem_t_t<tune_key::wg_tile_shape, shape<wg_tile_n, wg_tile_m>>>;
    using gemm_op_t = gpu::xetla::kernel::default_gemm_t<
            data_type_a, // input datatype for A
            mem_layout::row_major, // memory layout for A
            8, // leading dimension alignment for A, in unit of element
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for B
            8, // leading dimension alignment for B, in unit of element
            data_type_c, // output datatype for C
            mem_layout::row_major, // memory layout for C
            8, // leading dimension alignment for C, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            gpu_arch::Xe, // GPU arch
            tune_option>;

    // allocate temp buffers for global split
    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);
    auto Acc = alloc_device_and_init<data_type_acc>(
            size_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0.0f);
            },
            queue, device, context);
    auto Cnt = alloc_device_and_init<uint32_t>(
            size_cnt,
            [](uint32_t *data, size_t idx) {
                data[idx] = static_cast<uint32_t>(0);
            },
            queue, device, context);

    if constexpr (kslicing_type != kslicing_impl_t::none) {
        std::cout << "gemm_universal with "
                  << (kslicing_type == kslicing_impl_t::global ? "global"
                                                               : "local")
                  << " cooperation" << std::endl;
    }

    // set up gemm_universal arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n, Acc, Cnt);

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        free(A, context);
        free(B, context);
        free(C, context);
        free(Acc, context);
        free(Cnt, context);
        FAIL();
    }

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    //profiling_helper prof("gemm_universal", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) {
            //prof.cpu_start();
        }
        if constexpr (kslicing_type == kslicing_impl_t::global) {
            queue.memset(C, 0, size_c * sizeof(data_type_c));
        }
        queue.wait();
        typedef std::chrono::high_resolution_clock Clock;
        auto t0 = Clock::now();
        auto gpu_event = queue.submit([&](handler &cgh) {
            // GPU kernel
            cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                // allocate slm and nbarrier resource
                slm_barrier_init<gemm_op_t>();
                gemm_op_t gemm_op;
                gemm_op(item, gemm_arg);
            });
        });
        gpu_event.wait();
        queue.wait();
        auto t1 = Clock::now();
        float tt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        std::cout << tt << std::endl;


        if (i >= warmup) {
            //prof.cpu_end();
            //prof.add_gpu_event(gpu_event);
        }
    }

    ASSERT_EQ(0,
            gemm_result_validate(A, B, C, 1, matrix_m, matrix_k, matrix_n,
                    queue, mem_layout::row_major, mem_layout::row_major));

    //performance
    //prof.print_profiling_result(profiling_selector::GPU);

    free(A, context);
    free(B, context);
    free(C, context);
    free(Acc, context);
    free(Cnt, context);
}

template <kslicing_impl_t kslicing_type = kslicing_impl_t::none, typename DT, typename ACCT=float>
void gemm_universal_run_prealloc(uint32_t iter,
                                 sycl::queue &queue,
                                 sycl::context &context,
                                 sycl::device &device,
                                 DT* A, DT* B, DT* C,
                                 size_t matrix_m, size_t matrix_n, size_t matrix_k,
                                 bool check_correctness, bool cold_weights_cache,
                                 float &min_runtime, bool verbose=true) {

    if(verbose) {
        printf("===============================================\n");
        printf("benchmarking xetla(IPEX) mxk X kxn: (%zux%zu) x (%zux%zu)\n", matrix_m, matrix_k, matrix_k, matrix_n);
    }

    using data_type_a = DT;
    using data_type_b = DT;
    using data_type_c = DT;
    using data_type_acc = ACCT;

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("gemm_xetla_ipex", ops, "gflops");


    //Define the shape of workgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m
            = (kslicing_type != kslicing_impl_t::local) ? 256 : 64;
    constexpr uint32_t wg_tile_n
            = (kslicing_type != kslicing_impl_t::local) ? 256 : 128;

    // specify the range k_w/k_s by setting the corresponding ratio
    // splitk using global memory
    constexpr uint32_t num_global_splitk
            = (kslicing_type == kslicing_impl_t::global) ? 2 : 1;
    // splitk using local memory
    constexpr uint32_t num_local_splitk
            = (kslicing_type == kslicing_impl_t::local) ? 2 : 1;

    // Mirco-kernel configuration
    using tune_option = dict_t<
            elem_v_t<tune_key::param_optimizer_type,
                    tune_key_value::param_optimizer_decision_tree>,
            elem_t_t<tune_key::data_type_acc, data_type_acc>,
            elem_v_t<tune_key::dispatch_policy,
                    tune_key_value::dispatch_policy_kslicing>,
            elem_v_t<tune_key::global_kslicing_ratio, num_global_splitk>,
            elem_v_t<tune_key::local_kslicing_ratio, num_local_splitk>,
            elem_t_t<tune_key::wg_tile_shape, shape<wg_tile_n, wg_tile_m>>>;
    using gemm_op_t = gpu::xetla::kernel::default_gemm_t<
            data_type_a, // input datatype for A
            mem_layout::row_major, // memory layout for A
            8, // leading dimension alignment for A, in unit of element
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for B
            8, // leading dimension alignment for B, in unit of element
            data_type_c, // output datatype for C
            mem_layout::row_major, // memory layout for C
            8, // leading dimension alignment for C, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            gpu_arch::Xe, // GPU arch
            tune_option>;

    // allocate temp buffers for global split
    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);
    auto Acc = alloc_device_and_init<data_type_acc>(
            size_acc,
            [](data_type_acc *data, size_t idx) {
                data[idx] = static_cast<data_type_acc>(0.0f);
            },
            queue, device, context);
    auto Cnt = alloc_device_and_init<uint32_t>(
            size_cnt,
            [](uint32_t *data, size_t idx) {
                data[idx] = static_cast<uint32_t>(0);
            },
            queue, device, context);

    if constexpr (kslicing_type != kslicing_impl_t::none) {
        if(verbose) {
            std::cout << "gemm_universal with "
                      << (kslicing_type == kslicing_impl_t::global ? "global"
                                                                   : "local")
                      << " cooperation" << std::endl;
        }
    }

    // set up gemm_universal arguments
    typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
            matrix_k, B, matrix_n, C, matrix_n, Acc, Cnt);

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    if (!gemm_op_t::can_implement(gemm_arg)) {
        std::cout << "The arguments cannot be supported, aborting ... "
                  << std::endl;
        //free(A, context);
        //free(B, context);
        //free(C, context);
        free(Acc, context);
        free(Cnt, context);
        FAIL();
    }

    //reset global benchmarking per problem //TODO: wtf
    done_warmup = false;
    allevents.clear();
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if(cold_weights_cache) {
            DT* cold_b_ptr = fill_device_ptr<data_type_b>(queue, context, device, matrix_k * matrix_n);
            queue.memcpy(B, cold_b_ptr, matrix_k*matrix_n*sizeof(DT)).wait();
            free(cold_b_ptr, context);
        }
        queue.wait();

        if (i >= warmup) {
            done_warmup=true;
            prof.cpu_start();
        }

        {
            if constexpr (kslicing_type == kslicing_impl_t::global) {
                //queue.memset(C, 0, size_c * sizeof(data_type_c));
            }
            typedef std::chrono::high_resolution_clock Clock;
            auto t0 = Clock::now();
            auto gpu_event = queue.submit([&](handler &cgh) {
                // GPU kernel
                cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
                    // allocate slm and nbarrier resource
                    slm_barrier_init<gemm_op_t>();
                    gemm_op_t gemm_op;
                    gemm_op(item, gemm_arg);
                });
            });
            gpu_event.wait();
            //queue.wait();
        }

        if (i >= warmup) {
            prof.cpu_end();
            //prof.add_gpu_event(gpu_event);
        }
    }

    for(auto &e: allevents) {
        prof.add_gpu_event(e);
    }

    if(check_correctness) {
        ASSERT_EQ(0,
            gemm_result_validate(A, B, C, 1, matrix_m, matrix_k, matrix_n,
                    queue, mem_layout::row_major, mem_layout::row_major));
    }

    // performance
    //prof.print_profiling_result(profiling_selector::CPU);

    min_runtime = prof.get_min_cpu_time();
    if(verbose){
        printf("min time %f (ms)\n", prof.get_min_cpu_time());
        printf("-----------------------------------------------\n");
    }
}

template <typename DT, typename ACCT=float>
void gemm_onednn_run_prealloc(uint32_t iter,
                              sycl::queue &queue,
                              sycl::context &context,
                              sycl::device &device,
                              DT* A, DT* B, DT* C,
                              size_t matrix_m, size_t matrix_n, size_t matrix_k,
                              bool check_correctness, bool cold_weights_cache,
                              float &min_runtime, bool verbose=true) {

    //// DNNL Matmul
    // Create execution dnnl::engine
    //dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::engine engine = dnnl::sycl_interop::make_engine(device, context);

    // Create dnnl::stream.
    //dnnl::stream engine_stream(engine);
    dnnl::stream engine_stream = dnnl::sycl_interop::make_stream(engine, queue);

    // Source (A), weights (B), and destination (C) matrix dimensions.
    dnnl::memory::dims a_dims = {matrix_m, matrix_k};
    dnnl::memory::dims b_dims = {matrix_k, matrix_n};
    dnnl::memory::dims c_dims = {matrix_m, matrix_n};
    dnnl::memory::data_type type = dnnl::memory::data_type::f16;

    // Create memory descriptors and memory objects for src, weights, bias, and dst.
    auto a_md = dnnl::memory::desc(a_dims, type, dnnl::memory::format_tag::ab);
    auto b_md = dnnl::memory::desc(b_dims, type, dnnl::memory::format_tag::ab);
    auto c_md = dnnl::memory::desc(c_dims, type, dnnl::memory::format_tag::ab);

    auto a_in_md = dnnl::memory::desc(a_dims, type, dnnl::memory::format_tag::ab);
    auto b_in_md = dnnl::memory::desc(b_dims, type, dnnl::memory::format_tag::ab);
    auto a_in_mem = dnnl::memory(a_in_md, engine);
    auto b_in_mem = dnnl::memory(b_in_md, engine);

    uint8_t *a_ptr = (uint8_t *)a_in_mem.get_data_handle();
    uint8_t *b_ptr = (uint8_t *)b_in_mem.get_data_handle();

    size_t a_size = a_in_mem.get_desc().get_size();
    size_t b_size = b_in_mem.get_desc().get_size();

    queue.memcpy(a_ptr, A, a_size).wait();
    queue.memcpy(b_ptr, B, b_size).wait();

    // Create primitive descriptor.
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);

    // Repack and convert input data.
    auto a_mem = dnnl::memory(matmul_pd.src_desc(), engine);
    dnnl::reorder(a_in_mem, a_mem).execute(engine_stream, a_in_mem, a_mem);
    auto b_mem = dnnl::memory(matmul_pd.weights_desc(), engine);
    dnnl::reorder(b_in_mem, b_mem).execute(engine_stream, b_in_mem, b_mem);
    auto c_mem = dnnl::memory(matmul_pd.dst_desc(), engine);

    // Create the primitive.
    auto matmul_prim = dnnl::matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, dnnl::memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});


    if(verbose) {
        printf("===============================================\n");
        printf("ONEDNN benchmarking mxk X kxn: (%zux%zu) x (%zux%zu)\n", matrix_m, matrix_k, matrix_k, matrix_n);
    }

    using data_type_a = DT;
    using data_type_b = DT;
    using data_type_c = DT;
    using data_type_acc = ACCT;

    uint32_t warmup = 10;
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    profiling_helper prof("gemm_onednn", ops, "gflops");

    bool supported_config = true;
    //reset global benchmarking per problem //TODO: wtf
    done_warmup = false;
    allevents.clear();
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if(cold_weights_cache) {
            uint8_t *b_ptr = (uint8_t *)b_mem.get_data_handle();
            size_t b_size = b_mem.get_desc().get_size();

            data_type_b* cold_b_ptr = fill_device_ptr<data_type_b>(queue, context, device, b_size / sizeof(data_type_b));
            queue.memcpy(b_ptr, cold_b_ptr, b_size).wait();
            free(cold_b_ptr, context);
        }
        queue.wait();
        if (i >= warmup) {
            done_warmup = true;
            prof.cpu_start();
        }

        {
            matmul_prim.execute(engine_stream, matmul_args);
            engine_stream.wait();
        }


        if (i >= warmup) {
            prof.cpu_end();
            // prof.add_gpu_event(gpu_event);
        }
    }

    //for(auto &e: allevents) {
        //prof.add_gpu_event(e);
    //}

    ////
    ////

    if(check_correctness) {
        uint8_t *c_ptr = (uint8_t *)c_mem.get_data_handle();
        size_t c_size = c_mem.get_desc().get_size();
        queue.memcpy(C, c_ptr, c_size).wait();

        ASSERT_EQ(0,
                gemm_result_validate(A, B, C, 1, matrix_m, matrix_k, matrix_n,
                        queue, mem_layout::row_major, mem_layout::row_major));
    }

    // performance
    //prof.print_profiling_result(profiling_selector::CPU);

    min_runtime = prof.get_min_cpu_time();
    if(verbose) {
        printf("min time %f (ms)\n", prof.get_min_cpu_time());
        printf("-----------------------------------------------\n");
    }
}


//TODO: return
void benchmark_xetla_vs_onednn(sycl::queue &queue, sycl::context &context, sycl::device &device,
               keyMNK mnk,
               std::map<keyMNK, time_flops_bw> &perf_xetla,
               std::map<keyMNK, time_flops_bw> &perf_onednn,
               bool verbose=true) {

    size_t matrix_m = std::get<0>(mnk);
    size_t matrix_n = std::get<1>(mnk);
    size_t matrix_k = std::get<2>(mnk);

    //TODO: template DT
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
    using data_type_acc = float;

    data_type_a* d_A = fill_device_ptr<data_type_a>(queue, context, device, matrix_m * matrix_k);
    data_type_b* d_B = fill_device_ptr<data_type_a>(queue, context, device, matrix_k * matrix_n);
    data_type_c* d_C = fill_device_ptr<data_type_a>(queue, context, device, matrix_m * matrix_n);

    const int iter = 50;
    const bool check_correctness = false;
    const bool cold_weights_cache = false;
    float min_xetla_time, max_xetla_gflops, max_xetla_bw;
    gemm_universal_run_prealloc<kslicing_impl_t::local, data_type_a>(
            iter,
            queue, context, device,
            d_A, d_B, d_C,
            matrix_m, matrix_n, matrix_k,
            check_correctness, cold_weights_cache,
            min_xetla_time, verbose);
    long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;
    long bytes = (static_cast<long>(matrix_m) * matrix_k + matrix_k * matrix_n + matrix_m * matrix_n) * sizeof(data_type_a);
    max_xetla_gflops = (double)ops / min_xetla_time / 1000000;
    max_xetla_bw = (double)bytes / 1000000 / min_xetla_time ;
    perf_xetla[mnk] = {min_xetla_time, max_xetla_gflops, max_xetla_bw};
    if(verbose) {
        printf("GFlops %f GB/s %f \n", max_xetla_gflops, max_xetla_bw);
    }

    float min_onednn_time, max_onednn_gflops, max_onednn_bw;
    gemm_onednn_run_prealloc<data_type_a>(
            iter,
            queue, context, device,
            d_A, d_B, d_C,
            matrix_m, matrix_n, matrix_k,
            check_correctness, cold_weights_cache,
            min_onednn_time, verbose);
    max_onednn_gflops = (double)ops / min_onednn_time / 1000000;
    max_onednn_bw = (double)bytes / 1000000 / min_onednn_time ;
    perf_onednn[mnk] = {min_onednn_time, max_onednn_gflops, max_onednn_bw};
    if(verbose) {
        printf("GFlops %f GB/s %f \n", max_onednn_gflops, max_onednn_bw);
        printf("\n");
    }


    free(d_A, context);
    free(d_B, context);
    free(d_C, context);
}

void print_onednn_xetla_table(std::map<keyMNK, time_flops_bw> &perf_xetla,
                              std::map<keyMNK, time_flops_bw> &perf_onednn) {

    std::cout << "[M x N x K], xetla_min_time(ms), xetla_max_GFLOPS, xetla_max_BW(GB/s), onednn_min_time(ms), onednn_max_GFLOPS, onednn_max_BW(GB/s), onednn/xetla(%)" << std::endl;
    for(auto &kv : perf_xetla) {
        auto mnk = kv.first;
        std::cout << "[" << std::get<0>(mnk) << "x" <<
                     std::get<1>(mnk) << "x" <<
                     std::get<2>(mnk) << "], ";
        std::cout << std::get<0>(kv.second) << ", " << std::get<1>(kv.second) << ", " << std::get<2>(kv.second)  << ", ";
        std::cout << std::get<0>(perf_onednn[mnk])<< ", " << std::get<1>(perf_onednn[mnk]) << ", " << std::get<2>(perf_onednn[mnk]) << ", ";
        std::cout << std::get<0>(kv.second) / std::get<0>(perf_onednn[mnk]) * 100.f <<  "%" << std::endl;
    }
}

int main() {
    //Turn on the profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()};

    //Define SYCL queue, context and device
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    std::map<keyMNK, time_flops_bw> perf_xetla;
    std::map<keyMNK, time_flops_bw> perf_onednn;

    /*
    benchmark_xetla_vs_onednn(queue, context, device, {4, 12288, 4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096, 4096} , perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 16384, 4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096, 16384}, perf_xetla, perf_onednn);

    benchmark_xetla_vs_onednn(queue, context, device, {4, 11008, 4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096, 11008}, perf_xetla, perf_onednn);

    benchmark_xetla_vs_onednn(queue, context, device, {4, 15360, 5120}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120, 5120} , perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 13824, 5120}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120, 13824}, perf_xetla, perf_onednn);

    benchmark_xetla_vs_onednn(queue, context, device, {4, 16384, 4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50272, 4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 250880, 4096}, perf_xetla, perf_onednn);

    //M N K
    benchmark_xetla_vs_onednn(queue, context, device, {4, 12288, 4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096,  4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 16384, 4096}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096, 16384}, perf_xetla, perf_onednn);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50400, 4096}, perf_xetla, perf_onednn);
    */

    //benchmark_xetla_vs_onednn(queue, context, device, {4, 15360, 5120}, perf_xetla, perf_onednn);
    //benchmark_xetla_vs_onednn(queue, context, device, {4, 5120,  5120}, perf_xetla, perf_onednn);
    //benchmark_xetla_vs_onednn(queue, context, device, {4, 13824, 5120}, perf_xetla, perf_onednn);
    //benchmark_xetla_vs_onednn(queue, context, device, {4, 5120, 13824}, perf_xetla, perf_onednn);
    //benchmark_xetla_vs_onednn(queue, context, device, {4, 32000, 5120}, perf_xetla, perf_onednn);

    const bool verbose = false;

    /*
     * Tests for multi-device partitioning scaling
     */

    /// gpt-j-6b
    printf("****\n");
    printf("qkv_mm_fuse GPT-J-6b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 12288  , 4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 12288/2, 4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 12288/4, 4096}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_common GPT-J-6b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096,   4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096/2, 4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096/4, 4096}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_bias_gelu GPT-J-6b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 16384,   4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 16384/2, 4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 16384/4, 4096}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_bias_res_res GPT-J-6b K scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096, 16384},   perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096, 16384/2}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 4096, 16384/4}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("lmhead_mm GPT-J-6b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50400,   4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50400/2, 4096}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50400/4, 4096}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();
    printf("\n\n\n\n");
    /// /gpt-j-6b


    /// llama2-13b
    printf("****\n");
    printf("qkv_mm_fuse LLama2-13b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 15360  , 5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 15360/2, 5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 15360/4, 5120}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_common LLama2-13b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120,   5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120/2, 5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120/4, 5120}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_silu/resmul LLama2-13b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 13824,   5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 13824/2, 5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 13824/4, 5120}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_res LLama2-13b K scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120, 16384},   perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120, 16384/2}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 5120, 16384/4}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("lmhead_mm LLama2-13b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 32000,   5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 32000/2, 5120}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 32000/4, 5120}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();
    printf("\n\n\n\n");
    /// /llama2-13b


    /// bloom-176b
    printf("****\n");
    printf("qkv_mm_fuse bloom-176b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 10752  , 7168}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 10752/2, 7168}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 10752/4, 7168}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_common bloom-176b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 7168,   3584}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 7168/2, 3584}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 7168/4, 3584}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_bias_relu bloom-176b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 14336,   7168}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 14336/2, 7168}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 14336/4, 7168}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("mm_bias_res bloom-176b K scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 7168, 14336},   perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 7168, 14336/2}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 7168, 14336/4}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();

    printf("****\n");
    printf("lmhead_mm bloom-176b N scaling\n");
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50272,   2048}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50272/2, 2048}, perf_xetla, perf_onednn, verbose);
    benchmark_xetla_vs_onednn(queue, context, device, {4, 50272/4, 2048}, perf_xetla, perf_onednn, verbose);
    printf("\n*************\n");
    print_onednn_xetla_table(perf_xetla, perf_onednn);
    perf_xetla.clear(); perf_onednn.clear();
    printf("\n\n\n\n");
    /// /bloom-176b

    return (0);
}
