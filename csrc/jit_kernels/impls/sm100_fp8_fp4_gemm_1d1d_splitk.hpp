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

// JIT host for the single-kernel split-K FP8 block-scaled 1d1d GEMM that emits per-K-split partials.
// Grid = kNumSplits * num_m_blocks * num_n_blocks. Output d_partials is FP32 [kNumSplits, M, N].
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
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d_splitk.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_splitk_impl<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {}, {},
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
        args.num_splits,
        args.swizzle_a_mode, args.swizzle_b_mode, args.swizzle_cd_mode,
        args.num_stages,
        args.num_non_epilogue_threads, args.num_epilogue_threads,
        to_string(args.a_dtype), to_string(args.b_dtype), to_string(args.cd_dtype),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
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
    // Fixed tiles for the o_b decode shape (M<=128, N=4096, K=8192). BLOCK_M=128 covers all decode M.
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    constexpr int block_k = 128;
    constexpr int num_epilogue_threads = 128;          // one epilogue warp group (STORE_BLOCK_M=128)
    constexpr int num_non_epilogue_threads = 128;      // warps 0..3: TMA load / MMA / UTCCP

    // Only the 1-CTA, K-major, BF16-out-as-FP32-partial path is supported here.
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn and b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d_partials.scalar_type() == torch::kFloat);    // FP32 partials
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and sfb.scalar_type() == torch::kInt);  // packed UE8M0
    DG_HOST_ASSERT(num_splits >= 1);
    DG_HOST_ASSERT(n % block_n == 0);

    // HARD constraint: every split's k-offset must land on a packed-UE8M0 int32 boundary.
    // BLOCK_K * 4 = 512 elements per int32 (gran_k=128). Require K divisible by BLOCK_K*4*num_splits.
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);
    DG_HOST_ASSERT(k % (block_k * 4 * num_splits) == 0);

    // d_partials must be [num_splits, M, N], contiguous N-major per slice
    const auto [ns_, m_, n_] = get_shape<3>(d_partials);
    DG_HOST_ASSERT(ns_ == num_splits and m_ == m and n_ == n);

    const int num_m_blocks = ceil_div(m, block_m);
    const int num_n_blocks = ceil_div(n, block_n);

    const auto swizzle_a_mode  = get_swizzle_mode(block_k, a.element_size());
    const auto swizzle_b_mode  = get_swizzle_mode(block_k, b.element_size());
    const auto swizzle_cd_mode = get_swizzle_mode(block_n, d_partials.element_size());

    // A/B descriptors: identical to the non-split kernel (full [M,K] / [N,K], K-major).
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              block_m, block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              block_n, block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              swizzle_b_mode);

    // CD descriptor: 3D over [num_splits, M, N], FP32. (dim0=N inner, dim1=M, dim2=split)
    // Mirrors make_tma_3d_desc usage in the batched epilogue (sm100_fp8_bmm): smem dim0 is
    // overwritten to swizzle_cd_mode/elem_size inside make_tma_3d_desc, so we pass block_n here.
    const auto tensor_map_cd = make_tma_3d_desc(d_partials, n, m, num_splits,
                                                block_n, block_m, 1,
                                                static_cast<int>(d_partials.stride(-2)),
                                                static_cast<int>(d_partials.stride(-3)),
                                                swizzle_cd_mode);

    // SF descriptors: FULL (un-sliced) scales — kernel takes the per-split K slice via the SF TMA k-offset.
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 block_n, gran_k_b, 1, 0);

    // Compute stages from the SMEM budget (same layout as the base FP8 kernel).
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = align(block_n, 128);
    const int store_block_m = std::min(block_m, 128);
    const int smem_cd = store_block_m * swizzle_cd_mode * 2;   // STORE_BLOCK_M * kSwizzleCDMode * kNumTMAStoreStages
    int num_stages = 12, smem_size = 0;
    while (num_stages > 0) {
        const int smem_a_per_stage   = block_m * block_k * static_cast<int>(a.element_size());
        const int smem_b_per_stage   = block_n * block_k * static_cast<int>(b.element_size());
        const int smem_sfa_per_stage = sf_block_m * static_cast<int>(sizeof(uint32_t));
        const int smem_sfb_per_stage = sf_block_n * static_cast<int>(sizeof(uint32_t));
        const int smem_barriers = (num_stages * 3 + 2 /*kNumEpilogueStages*/ * 2) * 8;
        const int smem_tmem_ptr = 4;
        smem_size = smem_cd +
                    (smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage) * num_stages +
                    smem_barriers + smem_tmem_ptr;
        if (smem_size <= SM100ArchSpec::smem_capacity)
            break;
        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    const int grid_x = num_splits * num_m_blocks * num_n_blocks;

    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("[splitk] M:%d N:%d K:%d -> block %d/%d/%d, splits:%d, grid:%d, stages:%d, smem:%d, swz_cd:%d\n",
               m, n, k, block_m, block_n, block_k, num_splits, grid_x, num_stages, smem_size, swizzle_cd_mode);
    }

    const SM100FP8FP4Gemm1D1DSplitKRuntime::Args args = {
        .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .num_splits = num_splits,
        .swizzle_a_mode = swizzle_a_mode, .swizzle_b_mode = swizzle_b_mode, .swizzle_cd_mode = swizzle_cd_mode,
        .num_stages = num_stages,
        .num_non_epilogue_threads = num_non_epilogue_threads, .num_epilogue_threads = num_epilogue_threads,
        .gran_k_a = gran_k_a, .gran_k_b = gran_k_b,
        .major_a = major_a, .major_b = major_b,
        .compiled_dims = compiled_dims,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(), .cd_dtype = d_partials.scalar_type(),
        .epilogue_type = epilogue_type,
        .launch_args = LaunchArgs(grid_x, num_non_epilogue_threads + num_epilogue_threads, smem_size, 1),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100FP8FP4Gemm1D1DSplitKRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_gemm_1d1d_splitk", code);
    SM100FP8FP4Gemm1D1DSplitKRuntime::launch(runtime, args);
}

} // namespace deep_gemm
