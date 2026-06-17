#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// -------------------------------------------------------------------------------------------------
// Single-kernel split-K FP8 block-scaled (UE8M0) 1d1d GEMM that EMITS per-K-split partials.
//
// This is a self-contained, NON-persistent, 1-CTA (no 2-CTA multicast, no swap-AB) variant of
// `sm100_fp8_fp4_gemm_1d1d_impl`, with the K-routing modelled on `sm100_tf32_hc_prenorm_gemm_impl`.
//
//   grid.x  = kNumSplits * num_m_blocks * num_n_blocks
//   linear block index `bidx` decomposes as:
//       mn_block_idx = bidx / kNumSplits         (which is then split into m_block / n_block)
//       k_split_idx  = bidx % kNumSplits
//   Each CTA computes the partial GEMM over its own contiguous K-range only and stores its FP32
//   partial into d_partials[k_split_idx, :, :] via SM90_TMA_STORE_3D (NO internal reduction).
//   A downstream kernel (e.g. the mHC x-ring) sums the kNumSplits partials in FP32.
//
// HARD correctness constraint (enforced on the host): SHAPE_K % (BLOCK_K * 4 * kNumSplits) == 0.
//   -> every split's k_offset is a multiple of BLOCK_K * kNumSFAStagesPerLoad = 512 elements, so
//      the packed UE8M0 int32 SF boundary (4 scales / int32 at gran_k=128) is respected and the
//      LOCAL k_block_idx gate is identical to the GLOBAL one.
// -------------------------------------------------------------------------------------------------

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t,
          typename epilogue_type_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_fp4_gemm_1d1d_splitk_impl(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;

    // Partials are always FP32 (the MMA already applies the UE8M0 SFs inside, so the TMEM
    // accumulator is the dequantized FP32 partial; BF16 partials would lose precision).
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Split-K partials must be FP32");

    // MMA Configs (single CTA, no swap-AB)
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M, "Invalid block size");

    // SF configs
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
    DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    // Tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Utils
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_sfa_k = math::ceil_div(shape_k, kGranKA * 4);
    const auto shape_sfb_k = math::ceil_div(shape_k, kGranKB * 4);

    // ----- Split-K block routing (modelled on sm100_tf32_hc_prenorm_gemm_impl) -----
    const uint32_t num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t mn_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx  = block_idx % kNumSplits;
    const uint32_t m_block_idx  = mn_block_idx / num_n_blocks;
    const uint32_t n_block_idx  = mn_block_idx % num_n_blocks;

    constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t k_block_offset = k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks);
    const uint32_t num_total_k_blocks = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA/SFB shared memory
    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem  = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Wait for primary kernel completion (no-op for plain launches)
    cudaGridDependencySynchronize();

    // Pipeline and TMA phases (single (m,n,k_split) tile -> a single wave over the local K range)
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_local) {
        ++ k_local;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        for (uint32_t k_local = 0; k_local < num_total_k_blocks; advance_pipeline(k_local)) {
            // Wait consumer release
            empty_barriers[stage_idx]->wait(phase ^ 1);

            // GLOBAL k index for this split's local k-block
            const uint32_t k_idx = (k_block_offset + k_local) * BLOCK_K;
            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;

            // Issue A/B TMAs (K-major)
            DG_STATIC_ASSERT(kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K,
                             "Split-K variant only supports K-major A/B");
            tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx);
            tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx);
            auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE / (std::is_same_v<a_dtype_t, cutlass::float_e4m3_t> ? 1 : 2) +
                                     SMEM_B_SIZE_PER_STAGE / (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);

            // SF TMAs: gate on LOCAL k-block (== GLOBAL gate because k_block_offset % 4 == 0)
            if (k_local % kNumSFAStagesPerLoad == 0) {
                const uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                const uint32_t sfa_k_idx = math::ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad);
                tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
            }
            if (k_local % kNumSFBStagesPerLoad == 0) {
                const uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                const uint32_t sfb_k_idx = math::ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad);
                tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx);
                num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
            }

            full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
        }
    } else if (warp_idx == 1) {
        // MMA issue warp
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Single accumulator stage for this tile
        constexpr uint32_t accum_stage_idx = 0;
        tmem_empty_barriers[accum_stage_idx]->wait(1);
        ptx::tcgen05_after_thread_sync();

        auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
            if (do_tmem_full_arrive)
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
            __syncwarp();
        };

        using mma_t = ptx::SM100_MMA_MXF8F6F4_SS;
        for (uint32_t k_local = 0; k_local < num_total_k_blocks; advance_pipeline(k_local)) {
            with_sf_full_barriers[stage_idx]->wait(phase);
            ptx::tcgen05_after_thread_sync();

            const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
            const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
            if (cute::elect_one_sync()) {
                // Copy SF SMEM->TMEM at the start of each packed group (LOCAL gate)
                using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;
                const uint32_t sfa_stage_in_group_idx = k_local % kNumSFAStagesPerLoad;
                if (sfa_stage_in_group_idx == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                    }
                }
                const uint32_t sfb_stage_in_group_idx = k_local % kNumSFBStagesPerLoad;
                if (sfb_stage_in_group_idx == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                    }
                }

                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                    const uint32_t sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);
                    const uint32_t sfb_id = (kGranKB == 32 ? k : sfb_stage_in_group_idx);
                    const auto runtime_instr_desc =
                        mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                    a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                    b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                    // Accumulate predicate uses LOCAL k-block: each split starts its own accumulator fresh.
                    mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                               k_local > 0 or k > 0, runtime_instr_desc,
                               kTmemStartColOfSFA, kTmemStartColOfSFB);
                }
            }
            __syncwarp();

            // Last LOCAL k-block triggers the epilogue drain
            empty_barrier_arrive(k_local == num_total_k_blocks - 1);
        }
    } else if (warp_idx == 2) {
        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        for (uint32_t k_local = 0; k_local < num_total_k_blocks; advance_pipeline(k_local)) {
            full_barriers[stage_idx]->wait(phase);

            if (k_local % kNumSFAStagesPerLoad == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                    utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();
            }
            if (k_local % kNumSFBStagesPerLoad == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                    utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();
            }

            with_sf_full_barriers[stage_idx]->arrive(0u);
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp group
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        uint32_t tma_stage_idx = 0;
        constexpr uint32_t accum_stage_idx = 0;

        // Wait UMMA arrival
        tmem_full_barriers[accum_stage_idx]->wait(0);
        ptx::tcgen05_after_thread_sync();

        const auto tmem_base_addr = accum_stage_idx * UMMA_N;
        const auto base_m_idx = m_block_idx * BLOCK_M;
        const auto base_n_idx = n_block_idx * BLOCK_N;

        // Store this split's FP32 partial into d_partials[k_split_idx, :, :] via SM90_TMA_STORE_3D.
        // We reuse the existing epilogue's GemmType::Batched 3D-store branch with batch_idx = k_split_idx
        // and kWithAccumulation=false (-> SM90_TMA_STORE_3D, not REDUCE_ADD). Mainloop/scheduler are
        // plain (this is NOT the batched persistent path).
        epilogue::sm100_store_cd<
            BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
            kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
            GemmType::Batched, /*kWithAccumulation=*/false,
            cd_dtype_t, epilogue_type_t>
        (smem_cd, tma_stage_idx, tmem_base_addr,
         base_m_idx, base_n_idx, /*batch_idx=*/k_split_idx,
         epilogue_warp_idx, lane_idx,
         tmem_empty_barriers[accum_stage_idx],
         tensor_map_cd);
    }

    __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
