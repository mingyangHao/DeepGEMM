#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// JIT host for the persistent split-K FP8 block-scaled 1d1d GEMM that emits per-K-split partials.
// Output d_partials is FP32 or BF16 [kNumSplits, M, N].
// Scales (sfa/sfb) MUST already be in compute layout (int32-packed UE8M0, MN-major TMA-aligned) —
// this host does NOT call transform_sf_into_required_layout. Full (un-sliced) scales are passed;
// the per-split K slice is taken by the kernel via the SF TMA k-offset.
class SM100FP8FP4Gemm1D1DSplitKRuntime final: public LaunchRuntime<SM100FP8FP4Gemm1D1DSplitKRuntime> {
public:
    struct Args {
        int m, n, k;
        int block_m, block_n, block_k;
        int num_splits;
        int swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode;
        int num_stages;
        int num_non_epilogue_threads, num_epilogue_threads;
        int num_multicast;
        bool is_multicast_on_a;
        int num_sms;
        bool swap_ab;
        int gran_k_a, gran_k_b;
        cute::UMMA::Major major_a, major_b;
        std::string compiled_dims;
        at::ScalarType a_dtype, b_dtype, cd_dtype;
        std::optional<std::string> epilogue_type;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
        void* gmem_split_partials;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {},
        1,
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {},
        GemmType::Normal, false,
        {}, {}, {},
        {},
        {}
    >);
}};
)",
        to_string(args.major_a), to_string(args.major_b),
        args.gran_k_a, args.gran_k_b,
        get_compiled_dim(args.m, 'm', args.compiled_dims),
        get_compiled_dim(args.n, 'n', args.compiled_dims),
        get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.block_m, args.block_n, args.block_k,
        args.swizzle_a_mode, args.swizzle_b_mode, args.swizzle_cd_mode,
        args.num_stages,
        args.num_non_epilogue_threads, args.num_epilogue_threads,
        args.num_multicast, args.is_multicast_on_a ? "true" : "false",
        args.num_sms,
        args.swap_ab ? "true" : "false",
        to_string(args.a_dtype), to_string(args.b_dtype), to_string(args.cd_dtype),
        get_default_epilogue_type(args.epilogue_type),
        args.num_splits);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            static_cast<void*>(nullptr), args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd,
            args.gmem_split_partials));
    }
};

static void sm100_fp8_fp4_gemm_1d1d_splitk(const torch::Tensor& a, const torch::Tensor& sfa,
                                           const torch::Tensor& b, const torch::Tensor& sfb,
                                           const torch::Tensor& d_partials,
                                           const int& m, const int& n, const int& k,
                                           const int& num_splits,
                                           const int& gran_k_a, const int& gran_k_b,
                                           const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                           const std::string& compiled_dims,
                                           const std::optional<std::string>& epilogue_type = std::nullopt) {
    // Reuse the regular SM100 GEMM machinery (persistent scheduling and AB
    // swap) instead of maintaining a fixed-tile split kernel.
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn and b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d_partials.scalar_type() == torch::kFloat or
                   d_partials.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and sfb.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(num_splits > 1);
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d_partials.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    desc.check_validity();

    // Decode uses one logical M tile. Splitting that tile over K creates the
    // missing parallelism without duplicating the regular kernel's two M
    // tiles per split. Larger benchmark shapes use multiple BM=128 tiles so
    // the full M sweep measures actual split-K rather than dispatch fallback.
    // For DSV4-Pro N=7168 and M<=128, cluster-N=2 and split-K=2 yield
    // 56 * 2 = 112 useful CTA tasks in one wave.
    DG_HOST_ASSERT(ceil_div(n, 128) % 2 == 0 and desc.num_sms % 2 == 0);
    const auto layout = Layout {
        /*swap_ab=*/1,
        /*block_m=*/std::min(align(m, 16), 128),
        /*block_n=*/128,
        /*block_k=*/128,
        /*cluster_m=*/1,
        /*cluster_n=*/2
    };
    const auto storage = SM100ArchSpec::get_storage_config(desc, layout);
    const auto pipeline = SM100ArchSpec::get_pipeline_config(desc, layout, storage);
    const auto launch = SM100ArchSpec::get_launch_config(desc, layout);

    // Every partition must start on one packed UE8M0 int32 SF tile.
    DG_HOST_ASSERT(k % (layout.block_k * 4 * num_splits) == 0);

    // d_partials must be [num_splits, M, N], contiguous N-major per slice
    const auto [ns_, m_, n_] = get_shape<3>(d_partials);
    DG_HOST_ASSERT(ns_ == num_splits and m_ == m and n_ == n);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              storage.load_block_m, layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              storage.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              storage.load_block_n, layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              storage.swizzle_b_mode);

    // Split-K bypasses the regular TMA CD epilogue and writes TMEM
    // accumulators directly to d_partials. Keep a valid 2D descriptor because
    // the shared kernel prefetches it before dispatching warp roles.
    const auto tensor_map_cd = make_tma_cd_desc(d_partials, m, n,
                                                storage.store_block_m, storage.store_block_n,
                                                static_cast<int>(d_partials.stride(-2)), 1,
                                                storage.swizzle_cd_mode);

    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 layout.block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 layout.block_n, gran_k_b, 1, 0);

    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("[splitk-persistent] M:%d N:%d K:%d -> block %d/%d/%d, splits:%d, swap:%d, cluster:%d, stages:%d, smem:%d\n",
               m, n, k, layout.block_m, layout.block_n, layout.block_k,
               num_splits, layout.swap_ab, layout.get_cluster_size(),
               pipeline.num_stages, pipeline.smem_size);
    }

    const SM100FP8FP4Gemm1D1DSplitKRuntime::Args args = {
        .m = m, .n = n, .k = k,
        .block_m = layout.block_m, .block_n = layout.block_n, .block_k = layout.block_k,
        .num_splits = num_splits,
        .swizzle_a_mode = storage.swizzle_a_mode, .swizzle_b_mode = storage.swizzle_b_mode, .swizzle_cd_mode = storage.swizzle_cd_mode,
        .num_stages = pipeline.num_stages,
        .num_non_epilogue_threads = launch.num_non_epilogue_threads, .num_epilogue_threads = launch.num_epilogue_threads,
        .num_multicast = layout.get_cluster_size(), .is_multicast_on_a = layout.cluster_n > 1,
        .num_sms = launch.num_sms, .swap_ab = static_cast<bool>(layout.swap_ab),
        .gran_k_a = gran_k_a, .gran_k_b = gran_k_b,
        .major_a = major_a, .major_b = major_b,
        .compiled_dims = compiled_dims,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(), .cd_dtype = d_partials.scalar_type(),
        .epilogue_type = epilogue_type,
        .launch_args = LaunchArgs(launch.num_sms, launch.num_threads, pipeline.smem_size, layout.get_cluster_size()),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .gmem_split_partials = d_partials.data_ptr()
    };
    const auto code = SM100FP8FP4Gemm1D1DSplitKRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_gemm_1d1d_splitk", code);
    SM100FP8FP4Gemm1D1DSplitKRuntime::launch(runtime, args);
}

} // namespace deep_gemm
