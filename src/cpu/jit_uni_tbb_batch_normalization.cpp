/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <assert.h>

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_batch_normalization_utils.hpp"
#include "jit_generator.hpp"

#include "jit_uni_tbb_batch_normalization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

using namespace memory_tracking::names;
using namespace Xbyak;
typedef float data_t;

#define PARAM_ADDR(x) reg_param + offsetof(call_params_t, x)

enum { t0_pf_offt = 4096, t1_pf_offt = 0 };

template <cpu_isa_t isa>
struct jit_bnorm_fwd_t: public jit_generator {
    struct call_params_t {
        size_t N, C, S;
        const data_t *src, *dst;
        const uint8_t *ws;
        const data_t *mean, *var;
        const data_t *scale_shift;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_fwd_t);

    /* cpu specific part */
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
          isa == avx2, Ymm, Zmm>::type;
    const AddressFrame &vmmword =
        (isa == sse42) ? xword : (isa == avx2) ? yword : zword;

    enum {
        vlen = isa == sse42 ? 32 : cpu_isa_traits<isa>::vlen,
        simd_w = vlen / sizeof(data_t),
    };

    const batch_normalization_pd_t *bdesc_;

    void (*ker_)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker_)(p); }

    Reg64 reg_param = abi_param1;

    Reg64 reg_ptr_src = r15;
    Reg64 reg_ptr_dst = r14;
    Reg64 reg_ptr_ws = r13;
    Reg64 reg_ptr_mean = r12;
    Reg64 reg_ptr_var = r11;
    Reg64 reg_ptr_scale_shift = r10;

    Reg64 reg_off_dat_save = r9;
    Reg64 reg_off_dat = r8;
    Reg64 reg_off_c = rbx;

    Reg64 reg_N = rsi;
    Reg64 reg_C = rdx;
    Reg64 reg_S = rax;

    Reg64 reg_tmp = abi_not_param1;

    // Relu section
    bool with_relu, with_relu_inf_only;
    Opmask kstore_mask = Opmask(1);

    // channel tail processing
    Opmask ktail_mask = Opmask(2);

    Vmm vzero = Vmm(0);
    Vmm vone = Vmm(1);

    Vmm vmean = Vmm(2);
    Vmm vvar = Vmm(3);
    Vmm vsqrtvar = Vmm(4);

    Vmm vgamma = Vmm(5);
    Vmm vbeta = Vmm(6);

    Vmm veps = Vmm(7);

    Vmm vtmp = Vmm(14);
    Vmm v = Vmm(15);

    void load_common_params() {
#       define PARAM_PTR(x)  ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src, PARAM_PTR(src));
        mov(reg_ptr_dst, PARAM_PTR(dst));
        mov(reg_ptr_mean, PARAM_PTR(mean));
        mov(reg_ptr_var, PARAM_PTR(var));
        mov(reg_ptr_scale_shift, PARAM_PTR(scale_shift));
        mov(reg_ptr_ws, PARAM_PTR(ws));
#       undef PARAM_PTR

        Xmm x = Xmm(v.getIdx());

        mov(reg_tmp, float2int(bdesc_->desc()->batch_norm_epsilon));
        movq(x, reg_tmp);
        uni_vbroadcastss(veps, x);

        mov(reg_tmp, float2int(1.f));
        movq(x, reg_tmp);
        uni_vbroadcastss(vone, x);
    }

    void prepare_relu() {
        with_relu = bdesc_->is_fwd()
            ? bdesc_->with_relu_post_op() || bdesc_->fuse_bn_relu()
            : bdesc_->fuse_bn_relu();
        with_relu_inf_only = with_relu && bdesc_->is_fwd()
            && !(bdesc_->fuse_bn_relu() && bdesc_->is_training());

        if (with_relu) {
            uni_vpxor(vzero, vzero, vzero);
        }
    }

    void fwd_process_relu_avx2(Vmm vdst, Vmm vstore_mask) {
        Reg64 reg_store_mask = reg_tmp;
        shr(reg_off_dat, 5);
        vcmpps(vstore_mask, vzero, vdst, _cmp_lt_os);
        vmovmskps(reg_store_mask, vstore_mask);
        mov(ptr[reg_ptr_ws + reg_off_dat], reg_store_mask.cvt8());
        vblendvps(vdst, vzero, vdst, vstore_mask);
        shl(reg_off_dat, 5);
    }

    void fwd_process_relu_avx512_common(Vmm vdst) {
        shr(reg_off_dat, 5);
        vcmpps(kstore_mask, vzero, vdst, _cmp_lt_os);
        kmovw(ptr[reg_ptr_ws + reg_off_dat], kstore_mask);
        vblendmps(vdst | kstore_mask, vzero, vdst);
        shl(reg_off_dat, 5);
    }

    void forward(bool output_is_aligned) {
        const int stride_C = bdesc_->D() * bdesc_->H() * bdesc_->W() * simd_w;
        const int stride_N = (bdesc_->C() / simd_w) * stride_C;

        Label label_N, label_C, label_S;

        mov(reg_N, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save, reg_off_dat_save);
            xor_(reg_off_c, reg_off_c);

            mov(reg_C, dword[PARAM_ADDR(C)]);
            L(label_C);
            {
                mov(reg_off_dat, reg_off_dat_save);

                uni_vmovups(vmean, vmmword[reg_ptr_mean + reg_off_c]);
                uni_vmovups(vvar, vmmword[reg_ptr_var + reg_off_c]);

                uni_vmovups(vsqrtvar, vvar);
                uni_vaddps(vsqrtvar, vsqrtvar, veps);
                uni_vsqrtps(vsqrtvar, vsqrtvar);
                vdivps(vsqrtvar, vone, vsqrtvar);

                if (bdesc_->use_scaleshift()) {
                    uni_vmovups(vgamma, vmmword[reg_ptr_scale_shift + reg_off_c]);
                    int beta_off = bdesc_->C() * sizeof(data_t);
                    uni_vmovups(vbeta, vmmword[reg_ptr_scale_shift + reg_off_c + beta_off]);
                }

                mov(reg_S, dword[PARAM_ADDR(S)]);
                L(label_S);
                {
                    uni_vmovups(v, vmmword[reg_ptr_src + reg_off_dat]);
                    mic_prefetcht0(ptr[reg_ptr_src + reg_off_dat + t0_pf_offt]);
                    mic_prefetcht1(ptr[reg_ptr_src + reg_off_dat + t1_pf_offt]);
                    uni_vsubps(v, v, vmean);
                    uni_vmulps(v, v, vsqrtvar);

                    if (bdesc_->use_scaleshift())
                        uni_vfmadd213ps(v, vgamma, vbeta);

                    if (with_relu_inf_only) {
                        uni_vmaxps(v, v, vzero);
                    } else if (with_relu) {
                        if (isa == avx512_common)
                            fwd_process_relu_avx512_common(v);
                        else
                            fwd_process_relu_avx2(v, vtmp);
                    }

                    if (output_is_aligned) {
                        uni_vmovntps(vmmword[reg_ptr_dst + reg_off_dat], v);
                    } else {
                        uni_vmovups(vmmword[reg_ptr_dst + reg_off_dat], v);
                    }

                    add(reg_off_dat, simd_w * sizeof(data_t));

                    dec(reg_S);
                    jnz(label_S);
                }

                add(reg_off_dat_save, stride_C * sizeof(data_t));
                add(reg_off_c, simd_w * sizeof(data_t));

                dec(reg_C);
                jnz(label_C);
            }

            add(reg_ptr_src, stride_N * sizeof(data_t));
            add(reg_ptr_dst, stride_N * sizeof(data_t));
            add(reg_ptr_ws, stride_N / 8);

            dec(reg_N);
            jnz(label_N);
        }
    }

    jit_bnorm_fwd_t(const batch_normalization_pd_t *bdesc): bdesc_(bdesc) {
        static_assert(isa == avx2 || isa == avx512_common, "unsupported isa");

        preamble();
        load_common_params();
        prepare_relu();

        Label unaligned_store, end_store;
        test(reg_ptr_dst, vlen - 1);
        jnz(unaligned_store, T_NEAR);
        forward(true);
        jmp(end_store, T_NEAR);
        L(unaligned_store);
        forward(false);
        L(end_store);

        postamble();

        ker_ = getCode<decltype(ker_)>();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_bwd_t: public jit_generator {
    struct call_params_t {
        size_t N, C, S;
        const data_t *src, *diff_src, *diff_dst;
        const uint8_t *ws;
        const data_t *mean, *var;
        const data_t *scale_shift, *diff_scale_shift;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_bwd_t);

    /* cpu specific part */
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
          isa == avx2, Ymm, Zmm>::type;
    const AddressFrame &vmmword =
        (isa == sse42) ? xword : (isa == avx2) ? yword : zword;

    enum {
        vlen = isa == sse42 ? 32 : cpu_isa_traits<isa>::vlen,
        simd_w = vlen / sizeof(data_t),
    };

    const batch_normalization_pd_t *bdesc_;

    void (*ker_)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker_)(p); }

    Reg64 reg_param = abi_param1;

    Reg64 reg_ptr_src = r15;
    Reg64 reg_ptr_diff_src = r14;
    Reg64 reg_ptr_diff_dst = r13;
    Reg64 reg_ptr_ws = r12;

    Reg64 reg_ptr_c = r11;

    Reg64 reg_off_dat_save = r9;
    Reg64 reg_off_dat = r8;
    Reg64 reg_off_c = rbx;

    Reg64 reg_N = rsi;
    Reg64 reg_C = rdx;
    Reg64 reg_S = rax;

    Reg64 reg_tmp = abi_not_param1;

    // Relu section
    bool with_relu, with_relu_inf_only;
    Label l_relu_mask_avx2;
    Opmask kstore_mask = Opmask(1);

    // channel tail processing
    Opmask ktail_mask = Opmask(2);

    Vmm vzero = Vmm(0);
    Vmm vone = Vmm(1);

    Vmm vmean = Vmm(2);
    Vmm vsqrtvar = Vmm(3);

    Vmm vgamma = Vmm(4);
    Vmm vdiff_gamma = Vmm(5);
    Vmm vdiff_beta = Vmm(6);

    Vmm veps = Vmm(7);
    Vmm vNS = Vmm(8);

    Vmm vtmp = Vmm(14);
    Vmm v = Vmm(15);

    void load_common_params() {
#       define PARAM_PTR(x)  ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src, PARAM_PTR(src));
        mov(reg_ptr_diff_src, PARAM_PTR(diff_src));
        mov(reg_ptr_diff_dst, PARAM_PTR(diff_dst));
        mov(reg_ptr_ws, PARAM_PTR(ws));
#       undef PARAM_PTR

        Xmm x = Xmm(v.getIdx());

        mov(reg_tmp, float2int(bdesc_->desc()->batch_norm_epsilon));
        movq(x, reg_tmp);
        uni_vbroadcastss(veps, x);

        mov(reg_tmp, float2int(1.f));
        movq(x, reg_tmp);
        uni_vbroadcastss(vone, x);

        const int S = bdesc_->D() * bdesc_->H() * bdesc_->W();
        mov(reg_tmp, float2int(bdesc_->MB() * S));
        movq(x, reg_tmp);
        uni_vbroadcastss(vNS, x);
    }

    void prepare_relu() {
        with_relu = bdesc_->fuse_bn_relu();
        if (with_relu) {
            uni_vpxor(vzero, vzero, vzero);
            if (isa == avx2) prepare_l_relu_mask_avx2();
        }
    }

    void prepare_l_relu_mask_avx2() {
        Label l_mask_after;
        jmp(l_mask_after);
        align(32);
        L(l_relu_mask_avx2); /* [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01] */
        for (int i = 0; i < 8; ++i) dd(1<<i);
        L(l_mask_after);
    }

    void bwd_process_relu_avx2(Vmm vdiff_dst, Vmm vstore_mask) {
        shr(reg_off_dat, 5);
        vpbroadcastb(vstore_mask, ptr[reg_ptr_ws + reg_off_dat]);
        vpand(vstore_mask, vstore_mask, ptr[rip + l_relu_mask_avx2]);
        vpcmpeqd(vstore_mask, vstore_mask, ptr[rip + l_relu_mask_avx2]);
        vblendvps(vdiff_dst, vzero, vdiff_dst, vstore_mask);
        shl(reg_off_dat, 5);
    }

    void bwd_process_relu_avx512_common(Vmm vdiff_dst) {
        shr(reg_off_dat, 5);
        kmovw(kstore_mask, ptr[reg_ptr_ws + reg_off_dat]);
        vmovups(vdiff_dst | kstore_mask | T_z, vdiff_dst);
        shl(reg_off_dat, 5);
    }

    void load_c_specifics() {
        mov(reg_ptr_c, ptr[PARAM_ADDR(mean)]);
        uni_vmovups(vmean, vmmword[reg_ptr_c + reg_off_c]);

        mov(reg_ptr_c, ptr[PARAM_ADDR(var)]);
        uni_vmovups(vsqrtvar, vmmword[reg_ptr_c + reg_off_c]);
        uni_vaddps(vsqrtvar, vsqrtvar, veps);
        uni_vsqrtps(vsqrtvar, vsqrtvar);
        vdivps(vsqrtvar, vone, vsqrtvar);

        if (bdesc_->use_scaleshift()) {
            mov(reg_ptr_c, ptr[PARAM_ADDR(scale_shift)]);
            uni_vmovups(vgamma, vmmword[reg_ptr_c + reg_off_c]);
        }

        if (calculate_diff_stats()) {
            mov(reg_ptr_c, ptr[PARAM_ADDR(diff_scale_shift)]);
            uni_vmovups(vdiff_gamma, vmmword[reg_ptr_c + reg_off_c]);
            uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);
            uni_vdivps(vdiff_gamma, vdiff_gamma, vNS);
            int off = bdesc_->C() * sizeof(data_t);
            uni_vmovups(vdiff_beta, vmmword[reg_ptr_c + reg_off_c + off]);
            uni_vdivps(vdiff_beta, vdiff_beta, vNS);
        }
    }

    void backward(bool output_is_aligned) {
        const int stride_C = bdesc_->D() * bdesc_->H() * bdesc_->W() * simd_w;
        const int stride_N = (bdesc_->C() / simd_w) * stride_C;

        Label label_N, label_C, label_S;

        mov(reg_N, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save, reg_off_dat_save);
            xor_(reg_off_c, reg_off_c);

            mov(reg_C, dword[PARAM_ADDR(C)]);
            L(label_C);
            {
                mov(reg_off_dat, reg_off_dat_save);

                load_c_specifics();

                mov(reg_S, dword[PARAM_ADDR(S)]);
                L(label_S);
                {
                    uni_vmovups(v, vmmword[reg_ptr_diff_dst + reg_off_dat]);
                    if (with_relu) {
                        if (isa == avx512_common)
                            bwd_process_relu_avx512_common(v);
                        else if (isa == avx2)
                            bwd_process_relu_avx2(v, vtmp);
                        else
                            assert(false);
                    }

                    if (calculate_diff_stats()) {
                        uni_vsubps(v, v, vdiff_beta);
                        uni_vmovups(vtmp, vmmword[reg_ptr_src + reg_off_dat]);
                        uni_vsubps(vtmp, vtmp, vmean);
                        uni_vmulps(vtmp, vtmp, vdiff_gamma);
                        uni_vsubps(v, v, vtmp);
                    }

                    if (bdesc_->use_scaleshift())
                        uni_vmulps(v, v, vgamma);
                    uni_vmulps(v, v, vsqrtvar);

                    if (output_is_aligned) {
                        uni_vmovntps(vmmword[reg_ptr_diff_src + reg_off_dat], v);
                    } else {
                        uni_vmovups(vmmword[reg_ptr_diff_src + reg_off_dat], v);
                    }

                    mic_prefetcht0(ptr[reg_ptr_diff_dst + reg_off_dat + t0_pf_offt]);
                    mic_prefetcht0(ptr[reg_ptr_src + reg_off_dat + t0_pf_offt]);
                    mic_prefetcht1(ptr[reg_ptr_diff_dst + reg_off_dat + t1_pf_offt]);
                    mic_prefetcht1(ptr[reg_ptr_src + reg_off_dat + t1_pf_offt]);

                    add(reg_off_dat, simd_w * sizeof(data_t));

                    dec(reg_S);
                    jnz(label_S);
                }

                add(reg_off_dat_save, stride_C * sizeof(data_t));
                add(reg_off_c, simd_w * sizeof(data_t));

                dec(reg_C);
                jnz(label_C);
            }

            add(reg_ptr_src, stride_N * sizeof(data_t));
            add(reg_ptr_diff_src, stride_N * sizeof(data_t));
            add(reg_ptr_diff_dst, stride_N * sizeof(data_t));
            add(reg_ptr_ws, stride_N / 8);

            dec(reg_N);
            jnz(label_N);
        }
    }

    bool calculate_diff_stats() const { return !bdesc_->use_global_stats(); }

    jit_bnorm_bwd_t(const batch_normalization_pd_t *bdesc): bdesc_(bdesc) {
        static_assert(isa == avx2 || isa == avx512_common, "unsupported isa");

        preamble();
        load_common_params();
        prepare_relu();

        Label unaligned_store, end_store;
        test(reg_ptr_diff_src, vlen - 1);
        jnz(unaligned_store, T_NEAR);
        backward(true);
        jmp(end_store, T_NEAR);
        L(unaligned_store);
        backward(false);
        L(end_store);

        postamble();

        ker_ = getCode<decltype(ker_)>();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_bwd_diff_ss_t: public jit_generator {
    struct call_params_t {
        size_t N, C, S;
        const data_t *src, *diff_dst;
        const uint8_t *ws;
        const data_t *mean, *var;
        const data_t *diff_gamma, *diff_beta;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_bwd_diff_ss_t);

    /* cpu specific part */
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
          isa == avx2, Ymm, Zmm>::type;
    const AddressFrame &vmmword =
        (isa == sse42) ? xword : (isa == avx2) ? yword : zword;

    enum {
        vlen = isa == sse42 ? 32 : cpu_isa_traits<isa>::vlen,
        simd_w = vlen / sizeof(data_t),
    };

    const batch_normalization_pd_t *bdesc_;

    void (*ker_)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker_)(p); }

    Reg64 reg_param = abi_param1;

    Reg64 reg_ptr_src = r15;
    Reg64 reg_ptr_diff_dst = r14;
    Reg64 reg_ptr_ws = r13;

    Reg64 reg_ptr_diff_gamma = r12;
    Reg64 reg_ptr_diff_beta = r11;

    Reg64 reg_ptr_c = r10;

    Reg64 reg_off_dat_save = r9;
    Reg64 reg_off_dat = r8;
    Reg64 reg_off_c = rbx;

    Reg64 reg_N = rsi;
    Reg64 reg_C = rdx;
    Reg64 reg_S = rax;

    Reg64 reg_tmp = abi_not_param1;

    // Relu section
    bool with_relu, with_relu_inf_only;
    Label l_relu_mask_avx2;
    Opmask kstore_mask = Opmask(1);

    // channel tail processing
    Opmask ktail_mask = Opmask(2);

    Vmm vzero = Vmm(0);
    Vmm vone = Vmm(1);

    Vmm vmean = Vmm(2);
    Vmm vsqrtvar = Vmm(3);

    Vmm vgamma = Vmm(4);
    Vmm vdiff_gamma = Vmm(5);
    Vmm vdiff_beta = Vmm(6);

    Vmm veps = Vmm(7);

    Vmm vtmp = Vmm(14);
    Vmm v = Vmm(15);

    void load_common_params() {
#       define PARAM_PTR(x)  ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src, PARAM_PTR(src));
        mov(reg_ptr_diff_dst, PARAM_PTR(diff_dst));
        mov(reg_ptr_ws, PARAM_PTR(ws));
        mov(reg_ptr_diff_gamma, PARAM_PTR(diff_gamma));
        mov(reg_ptr_diff_beta, PARAM_PTR(diff_beta));
#       undef PARAM_PTR

        Xmm x = Xmm(v.getIdx());

        mov(reg_tmp, float2int(bdesc_->desc()->batch_norm_epsilon));
        movq(x, reg_tmp);
        uni_vbroadcastss(veps, x);

        mov(reg_tmp, float2int(1.f));
        movq(x, reg_tmp);
        uni_vbroadcastss(vone, x);
    }

    void prepare_relu() {
        with_relu = bdesc_->fuse_bn_relu();
        if (with_relu) {
            uni_vpxor(vzero, vzero, vzero);
            if (isa == avx2) prepare_l_relu_mask_avx2();
        }
    }

    void prepare_l_relu_mask_avx2() {
        Label l_mask_after;
        jmp(l_mask_after);
        align(32);
        L(l_relu_mask_avx2); /* [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01] */
        for (int i = 0; i < 8; ++i) dd(1<<i);
        L(l_mask_after);
    }

    void bwd_process_relu_avx2(Vmm vdiff_dst, Vmm vstore_mask) {
        shr(reg_off_dat, 5);
        vpbroadcastb(vstore_mask, ptr[reg_ptr_ws + reg_off_dat]);
        vpand(vstore_mask, vstore_mask, ptr[rip + l_relu_mask_avx2]);
        vpcmpeqd(vstore_mask, vstore_mask, ptr[rip + l_relu_mask_avx2]);
        vblendvps(vdiff_dst, vzero, vdiff_dst, vstore_mask);
        shl(reg_off_dat, 5);
    }

    void bwd_process_relu_avx512_common(Vmm vdiff_dst) {
        shr(reg_off_dat, 5);
        kmovw(kstore_mask, ptr[reg_ptr_ws + reg_off_dat]);
        vmovups(vdiff_dst | kstore_mask | T_z, vdiff_dst);
        shl(reg_off_dat, 5);
    }

    void load_c_specifics() {
        mov(reg_ptr_c, ptr[PARAM_ADDR(mean)]);
        uni_vmovups(vmean, vmmword[reg_ptr_c + reg_off_c]);

        mov(reg_ptr_c, ptr[PARAM_ADDR(var)]);
        uni_vmovups(vsqrtvar, vmmword[reg_ptr_c + reg_off_c]);
        uni_vaddps(vsqrtvar, vsqrtvar, veps);
        uni_vsqrtps(vsqrtvar, vsqrtvar);
        vdivps(vsqrtvar, vone, vsqrtvar);

        uni_vpxor(vdiff_gamma, vdiff_gamma, vdiff_gamma);
        uni_vpxor(vdiff_beta, vdiff_beta, vdiff_beta);
    }

    void backward() {
        const int stride_C = bdesc_->D() * bdesc_->H() * bdesc_->W() * simd_w;
        const int stride_N = (bdesc_->C() / simd_w) * stride_C;

        Label label_N, label_C, label_S;

        mov(reg_N, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save, reg_off_dat_save);
            xor_(reg_off_c, reg_off_c);

            mov(reg_C, dword[PARAM_ADDR(C)]);
            L(label_C);
            {
                mov(reg_off_dat, reg_off_dat_save);

                load_c_specifics();

                mov(reg_S, dword[PARAM_ADDR(S)]);
                L(label_S);
                {
                    uni_vmovups(v, vmmword[reg_ptr_diff_dst + reg_off_dat]);
                    if (with_relu) {
                        if (isa == avx512_common)
                            bwd_process_relu_avx512_common(v);
                        else if (isa == avx2)
                            bwd_process_relu_avx2(v, vtmp);
                        else
                            assert(false);
                    }

                    uni_vaddps(vdiff_beta, vdiff_beta, v);

                    uni_vmovups(vtmp, vmmword[reg_ptr_src + reg_off_dat]);
                    uni_vsubps(vtmp, vtmp, vmean);
                    uni_vfmadd231ps(vdiff_gamma, vtmp, v);

                    mic_prefetcht0(ptr[reg_ptr_diff_dst + reg_off_dat + t0_pf_offt]);
                    mic_prefetcht0(ptr[reg_ptr_src + reg_off_dat + t0_pf_offt]);
                    mic_prefetcht1(ptr[reg_ptr_diff_dst + reg_off_dat + t1_pf_offt]);
                    mic_prefetcht1(ptr[reg_ptr_src + reg_off_dat + t1_pf_offt]);

                    add(reg_off_dat, simd_w * sizeof(data_t));

                    dec(reg_S);
                    jnz(label_S);
                }

                uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);

                uni_vaddps(vdiff_gamma, vmmword[reg_ptr_diff_gamma + reg_off_c]);
                uni_vaddps(vdiff_beta, vmmword[reg_ptr_diff_beta + reg_off_c]);
                uni_vmovups(vmmword[reg_ptr_diff_gamma + reg_off_c], vdiff_gamma);
                uni_vmovups(vmmword[reg_ptr_diff_beta + reg_off_c], vdiff_beta);

                add(reg_off_dat_save, stride_C * sizeof(data_t));
                add(reg_off_c, simd_w * sizeof(data_t));

                dec(reg_C);
                jnz(label_C);
            }

            add(reg_ptr_src, stride_N * sizeof(data_t));
            add(reg_ptr_diff_dst, stride_N * sizeof(data_t));
            add(reg_ptr_ws, stride_N / 8);

            dec(reg_N);
            jnz(label_N);
        }
    }

    jit_bnorm_bwd_diff_ss_t(const batch_normalization_pd_t *bdesc): bdesc_(bdesc) {
        static_assert(isa == avx2 || isa == avx512_common, "unsupported isa");

        preamble();
        load_common_params();
        prepare_relu();
        backward();
        postamble();

        ker_ = getCode<decltype(ker_)>();
    }
};

template <cpu_isa_t isa>
struct uni_bnorm_driver_t: public c_compatible {
private:
    enum {
        simd_w = isa == sse42 ? 8 : cpu_isa_traits<isa>::vlen / sizeof(data_t)
    };

    struct bnorm_dims_t { int N, C, S; int glob; };

public:
    uni_bnorm_driver_t(const batch_normalization_pd_t *bdesc): bdesc_(bdesc)
    {
        nthr_ = mkldnn_get_max_threads();
        N_ = bdesc_->MB();
        S_ = bdesc_->D() * bdesc_->H() * bdesc_->W();
        C_blks_ = get_c_padded(bdesc_) / simd_w;

        const size_t l3_size = get_cache_size(3, true) * nthr_ / 2;
        const size_t working_set_size = sizeof(data_t) * N_ * S_ * simd_w;

        // to mimic jit_uni_bnorm thread distribution
        do_blocking_ = working_set_size * C_blks_ >= l3_size / 2 && l3_size > 0;

        C_blk_step_ = (int)(l3_size / working_set_size);
        C_blk_step_ = nstl::max(C_blk_step_, 1);
        C_blk_step_ = nstl::min(C_blk_step_, C_blks_);

        if (bdesc_->is_fwd()) {
            ker_fwd_ = new jit_bnorm_fwd_t<isa>(bdesc_);
        } else {
            ker_bwd_ = new jit_bnorm_bwd_t<isa>(bdesc_);
            ker_bwd_diff_ss_ = new jit_bnorm_bwd_diff_ss_t<isa>(bdesc_);
        }
    }
    ~uni_bnorm_driver_t()
    { delete ker_fwd_; delete ker_bwd_; delete ker_bwd_diff_ss_; }

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *bdesc) {
        int nthrs = mkldnn_get_max_threads();
        int C_PADDED = get_c_padded(bdesc);

        int sbuf_sz = use_tmp_stats(bdesc) * 2 * C_PADDED;
        int pbuf_sz = use_tmp_diff_scale_shift(bdesc) * 2 * C_PADDED;
        int rbuf_sz = (bdesc->is_fwd() ? 1 : 2) * C_PADDED * nthrs;

        scratchpad.book(key_bnorm_tmp_stats, sizeof(data_t) * sbuf_sz);
        scratchpad.book(key_bnorm_tmp_diff_ss, sizeof(data_t) * pbuf_sz);
        scratchpad.book(key_bnorm_reduction, sizeof(data_t) * rbuf_sz);
    }

    void exec_fwd_step_stats(const int C_blks, const bnorm_dims_t &nthr,
            const data_t *src, data_t *mean, data_t *var, data_t *rbuf) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;

        const int size_C_stat = C_blks * simd_w;

        const int nthr_NS = nthr.N * nthr.S;
        const bool need_reduction = nthr_NS > 1;

        auto reduce = [&](data_t *stat, data_t *r_stat) {
            if (!need_reduction) return;

#if 1
            data_t *loc_stat = r_stat;

            for (int c = 0; c < size_C_stat; ++c)
                stat[c] = loc_stat[c];

            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_stat += size_C_stat;
                for (int c = 0; c < size_C_stat; ++c)
                    stat[c] += loc_stat[c];
            }

            for (int c = 0; c < size_C_stat; ++c)
                stat[c] /= N_ * S_;
#else
            parallel(nthr.glob, [&](int ithr_glob, int) {
                const auto ithr = map_thread(ithr_glob, nthr);
                if (ithr.N > 0 || ithr.S > 0) return;

                int start_c, stop_c;
                work_distribution_c(C_blks, ithr.C, nthr.C, start_c, stop_c);

                data_t *loc_stat = r_stat;

                for (int c = start_c; c < stop_c; ++c)
                for (int simd = 0; simd < simd_w; ++simd)
                    stat[c * simd_w + simd] = loc_stat[c * simd_w + simd];

                for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                    loc_stat += size_C_stat;
                    for (int c = start_c; c < stop_c; ++c)
                    for (int simd = 0; simd < simd_w; ++simd)
                        stat[c * simd_w + simd] += loc_stat[c * simd_w + simd];
                }

                for (int c = start_c; c < stop_c; ++c)
                for (int simd = 0; simd < simd_w; ++simd)
                    stat[c * simd_w + simd] /= N_ * S_;
            });
#endif
        };

        data_t *r_mean = need_reduction ? rbuf : mean;

        // find local mean
        parallel(nthr.glob, [&](int ithr_glob, int) {
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            data_t *loc_mean = &r_mean[ithr_NS * size_C_stat];

            for (int c = start.C; c < stop.C; ++c)
            for (int simd = 0; simd < simd_w; ++simd)
                loc_mean[c * simd_w + simd] = 0;

            for (int n = start.N; n < stop.N; ++n)
            for (int c = start.C; c < stop.C; ++c)
            for (int s = start.S; s < stop.S; ++s)
            for (int simd = 0; simd < simd_w; ++simd)
            {
                loc_mean[c * simd_w + simd] +=
                    src[n * stride_N + c * stride_C + s * simd_w + simd];
            }

            if (!need_reduction) {
                for (int c = start.C; c < stop.C; ++c)
                for (int simd = 0; simd < simd_w; ++simd)
                    loc_mean[c * simd_w + simd] /= N_ * S_;
            }
        });

        // mean reduction
        reduce(mean, r_mean);

        data_t *r_var = need_reduction ? rbuf : var;

        // find local var
        parallel(nthr.glob, [&](int ithr_glob, int) {
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            data_t *loc_var = &r_var[ithr_NS * size_C_stat];

            for (int c = start.C; c < stop.C; ++c)
            for (int simd = 0; simd < simd_w; ++simd)
                loc_var[c * simd_w + simd] = 0;

            for (int n = start.N; n < stop.N; ++n)
            for (int c = start.C; c < stop.C; ++c)
            for (int s = start.S; s < stop.S; ++s)
            for (int simd = 0; simd < simd_w; ++simd)
            {
                data_t src_v =
                    src[n * stride_N + c * stride_C + s * simd_w + simd];
                data_t v = src_v - mean[c * simd_w + simd];

                loc_var[c * simd_w + simd] += v * v;
            }

            if (!need_reduction) {
                for (int c = start.C; c < stop.C; ++c)
                for (int simd = 0; simd < simd_w; ++simd)
                    loc_var[c * simd_w + simd] /= N_ * S_;
            }
        });

        // var reduction
        reduce(var, r_var);
    }

    void exec_fwd_step_normalization(const int C_blks, const bnorm_dims_t &nthr,
            const data_t *src, data_t *dst, const data_t *scale_shift,
            const data_t *mean, const data_t *var, uint8_t *ws) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;

        parallel(nthr.glob, [&](int ithr_glob, int) {
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

#if 1
            auto c = typename jit_bnorm_fwd_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off =
                start.N * stride_N + start.C * stride_C + start.S * simd_w;
            c.src = &src[d_off];
            c.dst = &dst[d_off];
            c.ws = &ws[d_off / 8];
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.scale_shift = &scale_shift[start.C * simd_w];

            (*ker_fwd_)(&c);
#else
            const data_t eps = bdesc_->desc()->batch_norm_epsilon;
            for (int n = start.N; n < stop.N; ++n)
            for (int c = start.C; c < stop.C; ++c)
            for (int s = start.S; s < stop.S; ++s)
            for (int simd = 0; simd < simd_w; ++simd)
            {
                const size_t d_off =
                    n * stride_N + c * stride_C + s * simd_w + simd;
                data_t sqrt_variance =
                    1.f / sqrtf(var[c * simd_w + simd] + eps);
                data_t res =
                    scale_shift[c * simd_w + simd]
                    * (src[d_off] - mean[c * simd_w + simd])
                    * sqrt_variance
                    + scale_shift[(C_blks_ + c) * simd_w + simd];

                dst[d_off] = res;
            }
#endif
        });
    }

    void exec_fwd(const data_t *src, data_t *dst, const data_t *scale_shift,
            data_t *mean, data_t *var, uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto rbuf = scratchpad.get<data_t>(key_bnorm_reduction);
        if (use_tmp_stats(bdesc_)) {
            auto sbuf = scratchpad.get<data_t>(key_bnorm_tmp_stats);
            mean = sbuf;
            var = sbuf + C_blks_ * simd_w;
        }

        const size_t stride_C = (size_t)S_ * simd_w;

        int C_blk_step = C_blk_step_;
        auto nthr = bnorm_dims_t();

        thread_distribution(C_blk_step, nthr);

        for (int C_blk_st = 0; C_blk_st < C_blks_; C_blk_st += C_blk_step) {
            if (C_blk_st + C_blk_step > C_blks_) {
                C_blk_step = C_blks_ - C_blk_st;
                thread_distribution(C_blk_step, nthr);
            }

            if (!bdesc_->stats_is_src()) {
                exec_fwd_step_stats(
                        C_blk_step,
                        nthr,
                        src + C_blk_st * stride_C,
                        mean + C_blk_st * simd_w,
                        var + C_blk_st * simd_w,
                        rbuf);
            }

            exec_fwd_step_normalization(
                    C_blk_step,
                    nthr,
                    src + C_blk_st * stride_C,
                    dst + C_blk_st * stride_C,
                    scale_shift + C_blk_st * simd_w,
                    mean + C_blk_st * simd_w,
                    var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / 8);
        }
    }

    void exec_bwd_step_diff_ss(const int C_blks, const bnorm_dims_t &nthr,
            const data_t *src, const data_t *diff_dst, const data_t *mean,
            const data_t *var, const uint8_t *ws, data_t *diff_ss,
            data_t *rbuf) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;

        const int size_C_stat = C_blks * simd_w;

        const int nthr_NS = nthr.N * nthr.S;
        const bool need_reduction = nthr_NS > 1;

        data_t *diff_gamma = diff_ss;
        data_t *diff_beta = diff_ss + C_blks_ * simd_w;

        data_t * const r_diff_gamma = need_reduction ? rbuf : diff_gamma;
        data_t * const r_diff_beta = need_reduction
            ? rbuf + nthr_NS * size_C_stat : diff_beta;

        auto reduce = [&]() {
            if (!need_reduction) return;

#if 1
            // diff_gamma
            const data_t *loc_diff_gamma = r_diff_gamma;
            for (int c = 0; c < size_C_stat; ++c)
                diff_gamma[c] = loc_diff_gamma[c];
            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_diff_gamma += size_C_stat;
                for (int c = 0; c < size_C_stat; ++c)
                    diff_gamma[c] += loc_diff_gamma[c];
            }

            // diff_beta
            const data_t *loc_diff_beta = r_diff_beta;
            for (int c = 0; c < size_C_stat; ++c)
                diff_beta[c] = loc_diff_beta[c];
            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_diff_beta += size_C_stat;
                for (int c = 0; c < size_C_stat; ++c)
                    diff_beta[c] += loc_diff_beta[c];
            }
#else
            parallel(nthr.glob, [&](int ithr_glob, int) {
                const auto ithr = map_thread(ithr_glob, nthr);
                if (ithr.N > 0 || ithr.S > 0) return;

                int start_c, stop_c;
                work_distribution_c(C_blks, ithr.C, nthr.C, start_c, stop_c);

                // diff_gamma
                const data_t *loc_diff_gamma = r_diff_gamma;
                for (int c = start_c * simd_w; c < stop_c * simd_w; ++c)
                    diff_gamma[c] = loc_diff_gamma[c];
                for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                    loc_diff_gamma += size_C_stat;
                    for (int c = start_c * simd_w; c < stop_c * simd_w; ++c)
                        diff_gamma[c] += loc_diff_gamma[c];
                }

                // diff_beta
                const data_t *loc_diff_beta = r_diff_beta;
                for (int c = start_c * simd_w; c < stop_c * simd_w; ++c)
                    diff_beta[c] = loc_diff_beta[c];
                for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                    loc_diff_beta += size_C_stat;
                    for (int c = start_c * simd_w; c < stop_c * simd_w; ++c)
                        diff_beta[c] += loc_diff_beta[c];
                }
            });
#endif
        };

        parallel(nthr.glob, [&](int ithr_glob, int) {
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            data_t *loc_diff_gamma = &r_diff_gamma[ithr_NS * size_C_stat];
            data_t *loc_diff_beta = &r_diff_beta[ithr_NS * size_C_stat];

            for (int c = start.C; c < stop.C; ++c)
            for (int simd = 0; simd < simd_w; ++simd)
                loc_diff_gamma[c * simd_w + simd] = 0;

            for (int c = start.C; c < stop.C; ++c)
            for (int simd = 0; simd < simd_w; ++simd)
                loc_diff_beta[c * simd_w + simd] = 0;

#if 1
            auto c = typename jit_bnorm_bwd_diff_ss_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off =
                start.N * stride_N + start.C * stride_C + start.S * simd_w;
            c.src = &src[d_off];
            c.diff_dst = &diff_dst[d_off];
            c.ws = &ws[d_off / 8];
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.diff_gamma = &loc_diff_gamma[start.C * simd_w];
            c.diff_beta = &loc_diff_beta[start.C * simd_w];

            (*ker_bwd_diff_ss_)(&c);
#else
            for (int n = start.N; n < stop.N; ++n)
            for (int c = start.C; c < stop.C; ++c)
            for (int s = start.S; s < stop.S; ++s)
            for (int simd = 0; simd < simd_w; ++simd)
            {
                const size_t d_off
                    = n * stride_N + c * stride_C + s * simd_w + simd;
                const size_t c_off = (c - start.C) * simd_w + simd;

                loc_diff_gamma[c_off]
                    += (src[d_off] - mean[c * simd_w + simd]) * diff_dst[d_off];
                loc_diff_beta[c_off] += diff_dst[d_off];
            }

            const data_t eps = bdesc_->desc()->batch_norm_epsilon;
            for (int c = start.C; c < stop.C; ++c)
            for (int simd = 0; simd < simd_w; ++simd)
                loc_diff_gamma[(c - start.C) * simd_w + simd]
                    /= sqrtf(var[c * simd_w + simd] + eps);
#endif
        });

        reduce();
    }

    void exec_bwd_step_normalization(const int C_blks, const bnorm_dims_t &nthr,
            const data_t *src, data_t *diff_src, const data_t *diff_dst,
            const data_t *mean, const data_t *var,
            const uint8_t *ws,
            const data_t *scale_shift, const data_t *diff_ss) {
        const size_t stride_C = (size_t)S_ * simd_w;
        const size_t stride_N = (size_t)C_blks_ * stride_C;

        parallel(nthr.glob, [&](int ithr_glob, int) {
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

#if 1
            auto c = typename jit_bnorm_bwd_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off =
                start.N * stride_N + start.C * stride_C + start.S * simd_w;
            c.src = &src[d_off];
            c.diff_src = &diff_src[d_off];
            c.diff_dst = &diff_dst[d_off];
            c.ws = &ws[d_off / 8];
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.scale_shift = &scale_shift[start.C * simd_w];
            c.diff_scale_shift = &diff_ss[start.C * simd_w];

            (*ker_bwd_)(&c);
#else
            const float NS = N_ * S_;
            const bool calculate_diff_stats = !bdesc_->use_global_stats();
            const data_t eps = bdesc_->desc()->batch_norm_epsilon;

            const data_t *diff_gamma = diff_ss;
            const data_t *diff_beta = diff_ss + C_blks_ * simd_w;

            for (int n = start.N; n < stop.N; ++n)
            for (int c = start.C; c < stop.C; ++c)
            for (int s = start.S; s < stop.S; ++s)
            for (int simd = 0; simd < simd_w; ++simd)
            {
                const size_t d_off =
                    n * stride_N + c * stride_C + s * simd_w + simd;
                const size_t c_off = c * simd_w + simd;

                data_t sqrt_variance = 1.f / sqrtf(var[c_off] + eps);
                data_t res = diff_dst[d_off];

                if (calculate_diff_stats) {
                    res +=
                        - diff_beta[c_off] / NS
                        - (src[d_off] - mean[c_off])
                        * diff_gamma[c_off] * sqrt_variance / NS;
                }

                diff_src[d_off]
                    = res * scale_shift[c * simd_w + simd] * sqrt_variance;
            }
#endif
        });
    }

    void exec_bwd(const data_t *src, data_t *diff_src, const data_t *diff_dst,
            const data_t *scale_shift, data_t *diff_scale_shift,
            const data_t *mean, const data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto rbuf = scratchpad.get<data_t>(key_bnorm_reduction);
        if (use_tmp_diff_scale_shift(bdesc_)) {
            auto pbuf = scratchpad.get<data_t>(key_bnorm_tmp_diff_ss);
            diff_scale_shift = pbuf;
        }

        const size_t stride_C = (size_t)S_ * simd_w;

        int C_blk_step = C_blk_step_;
        auto nthr = bnorm_dims_t();

        thread_distribution(C_blk_step, nthr);

        for (int C_blk_st = 0; C_blk_st < C_blks_; C_blk_st += C_blk_step) {
            if (C_blk_st + C_blk_step > C_blks_) {
                C_blk_step = C_blks_ - C_blk_st;
                thread_distribution(C_blk_step, nthr);
            }

            exec_bwd_step_diff_ss(
                    C_blk_step,
                    nthr,
                    src + C_blk_st * stride_C,
                    diff_dst + C_blk_st * stride_C,
                    mean + C_blk_st * simd_w,
                    var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / 8,
                    diff_scale_shift + C_blk_st * simd_w,
                    rbuf);

            exec_bwd_step_normalization(
                    C_blk_step,
                    nthr,
                    src + C_blk_st * stride_C,
                    diff_src + C_blk_st * stride_C,
                    diff_dst + C_blk_st * stride_C,
                    mean + C_blk_st * simd_w,
                    var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / 8,
                    scale_shift + C_blk_st * simd_w,
                    diff_scale_shift + C_blk_st * simd_w);
        }
    }

private:
    static bool use_tmp_stats(const batch_normalization_pd_t *bdesc) {
        return true
            && !bdesc->stats_is_src()
            && bdesc->desc()->prop_kind == prop_kind::forward_inference;
    }

    static bool use_tmp_diff_scale_shift(const batch_normalization_pd_t *bdesc)
    {
        return false
            || (bdesc->is_bwd() && !bdesc->use_scaleshift())
            || bdesc->desc()->prop_kind == prop_kind::backward_data;
    }

    static int get_c_padded(const batch_normalization_pd_t *bdesc)
    { return bdesc->src_pd()->desc()->layout_desc.blocking.padding_dims[1]; }

    void thread_distribution(int C_blks, bnorm_dims_t &nthr) {
        if (do_blocking_) {
            nthr.N = nstl::min(N_, nthr_);
            nthr.C = nstl::min(C_blks, nthr_ / nthr.N);
        } else {
            // nthr.C = nthr_ < C_blks ? nthr_ : math::gcd(nthr_, C_blks);
            nthr.C = math::gcd(nthr_, C_blks);
            nthr.N = nstl::max(1, nstl::min(N_, nthr_ / nthr.C));
        }
        nthr.S = nstl::max(1, nstl::min(S_, nthr_ / nthr.C / nthr.N));
        nthr.glob = nthr.N * nthr.C * nthr.S;
    }

    int map_thread_c(int ithr_glob, const bnorm_dims_t &nthr)
    { return ithr_glob / nthr.N / nthr.S; }

    bnorm_dims_t map_thread(int ithr_glob, const bnorm_dims_t &nthr) {
        auto ithr = bnorm_dims_t();
        ithr.glob = ithr_glob;
        ithr.C = map_thread_c(ithr.glob, nthr);
        ithr.N = ithr.glob / nthr.S % nthr.N;
        ithr.S = ithr.glob % nthr.S;
        return ithr;
    }

    void work_distribution_c(int C_blks, int ithr_c, int nthr_c,
            int &start_c, int &stop_c)
    { balance211(C_blks, nthr_c, ithr_c, start_c, stop_c); }

    void work_distribution(int C_blks,
            const bnorm_dims_t &ithr, const bnorm_dims_t &nthr,
            bnorm_dims_t &start, bnorm_dims_t &stop) {
        work_distribution_c(C_blks, ithr.C, nthr.C, start.C, stop.C);
        balance211(N_, nthr.N, ithr.N, start.N, stop.N);
        balance211(S_, nthr.S, ithr.S, start.S, stop.S);
    }

    const batch_normalization_pd_t *bdesc_;

    bool do_blocking_;  // mimics regular jit_uni_bnorm;
                        //defines thread distribution (C,N,S) vs (N,C,S)

    int nthr_;

    int N_, S_;         // MB, D * H *W
    int C_blks_;        // C / simd_w
    int C_blk_step_;    // for C_blks = 0 .. C_blks_, += C_blk_step_

    jit_bnorm_fwd_t<isa> *ker_fwd_ = nullptr;
    jit_bnorm_bwd_t<isa> *ker_bwd_ = nullptr;
    jit_bnorm_bwd_diff_ss_t<isa> *ker_bwd_diff_ss_ = nullptr;
};

}

using namespace data_type;
using namespace memory_format;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_fwd_t<isa>::pd_t::init() {
    assert(engine()->kind() == engine_kind::cpu);
    auto desired_fmt = ndims() == 4
        ? isa == avx512_common ? nChw16c : nChw8c
        : isa == avx512_common ? nCdhw16c : nCdhw8c;

    static int dont_want = -1;
    if (dont_want == -1) {
        const char *dont_want_str = getenv("BN");
        dont_want = dont_want_str && dont_want_str[0] == '0';
    }

    bool ok = true
        // && !mkldnn_thr_syncable()
        && !dont_want
        && mayiuse(isa)
        && is_fwd()
        && !has_zero_dim_memory()
        && one_of(ndims(), 4, 5)
        && desc()->data_desc.data_type == f32
        && IMPLICATION(use_scaleshift(),
                desc()->data_scaleshift_desc.data_type == f32)
        && desc()->data_desc.format == desired_fmt
        && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    if (is_training() && fuse_bn_relu()) {
        if (isa < avx2) return status::unimplemented;
        bn_init_default_ws(this, this->workspace_pd_, 1);
    }

    if (memory_desc_wrapper(&data_pd_).blocking_desc().padding_dims[1]
            != this->C() /* && isa < avx2 */)
        return status::unimplemented;

    if (stats_is_src() || is_training()) {
        memory_desc_t stats_d;
        dims_t stats_dims = { C() };
        mkldnn_memory_desc_init(&stats_d, 1, stats_dims, f32, x);
        mean_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
        variance_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
    }

    auto scratchpad = scratchpad_registry().registrar();
    uni_bnorm_driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_fwd_t<isa>::
jit_uni_tbb_batch_normalization_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs)
{ bnorm_driver_ = new uni_bnorm_driver_t<isa>(pd()); }

template <cpu_isa_t isa>
void jit_uni_tbb_batch_normalization_fwd_t<isa>::execute(event_t *e) const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    auto mean = reinterpret_cast<data_t *>(pd()->stats_is_src()
            ? const_cast<char *>(this->input_memory(1))
            : this->memory(1));
    auto var = reinterpret_cast<data_t *>(pd()->stats_is_src()
            ? const_cast<char *>(this->input_memory(2))
            : this->memory(2));

    auto idx_scale_shift = 1 + 2 * pd()->stats_is_src();
    auto scale_shift =
        reinterpret_cast<const data_t *>(this->input_memory(idx_scale_shift));
    auto ws = reinterpret_cast<uint8_t *>(this->memory(pd()->ws_idx()));

    auto scratchpad = this->scratchpad();

    bnorm_driver_->exec_fwd(src, dst, scale_shift, mean, var, ws, scratchpad);

    e->set_state(event_t::ready);
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_fwd_t<isa>::
~jit_uni_tbb_batch_normalization_fwd_t() { delete bnorm_driver_; }

/* struct instantiation */
template struct jit_uni_tbb_batch_normalization_fwd_t<avx2>;
template struct jit_uni_tbb_batch_normalization_fwd_t<avx512_common>;

/* bwd */

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_bwd_t<isa>::pd_t::init() {
    assert(engine()->kind() == engine_kind::cpu);
    auto desired_fmt = ndims() == 4
        ? isa == avx512_common ? nChw16c : nChw8c
        : isa == avx512_common ? nCdhw16c : nCdhw8c;

    static int dont_want = -1;
    if (dont_want == -1) {
        const char *dont_want_str = getenv("BN");
        dont_want = dont_want_str && dont_want_str[0] == '0';
    }

    bool ok = true
        // && !mkldnn_thr_syncable()
        && !dont_want
        && mayiuse(isa)
        && is_bwd()
        && !has_zero_dim_memory()
        && one_of(ndims(), 4, 5)
        && everyone_is(f32, desc()->data_desc.data_type,
                desc()->diff_data_desc.data_type)
        && IMPLICATION(use_scaleshift(),
                desc()->data_scaleshift_desc.data_type == f32)
        && everyone_is(desired_fmt, desc()->diff_data_desc.format,
                desc()->data_desc.format)
        && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    if (memory_desc_wrapper(&data_pd_).blocking_desc()
            .padding_dims[1] != this->C() /* && isa < avx2 */)
        return status::unimplemented;

    if (fuse_bn_relu()) {
        if (isa < avx2) return status::unimplemented;
        bn_init_default_ws(this, this->workspace_pd_, 1);
        size_t this_ws_sz = memory_desc_wrapper(this->workspace_pd()).size();

        bool ws_ok = true
            && hint_fwd_pd_->workspace_pd()
            && memory_desc_wrapper(hint_fwd_pd_->workspace_pd()).size()
            == this_ws_sz;
        if (!ws_ok) return status::unimplemented;
    }

    /* TODO: extra checks required */

    auto scratchpad = scratchpad_registry().registrar();
    uni_bnorm_driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_bwd_t<isa>::
jit_uni_tbb_batch_normalization_bwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs)
{ bnorm_driver_ = new uni_bnorm_driver_t<isa>(pd()); }

template <cpu_isa_t isa>
void jit_uni_tbb_batch_normalization_bwd_t<isa>::execute(event_t *e) const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto mean = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto var = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(3));
    auto scale_shift = reinterpret_cast<const data_t *>(this->input_memory(4));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_scale_shift = reinterpret_cast<data_t *>(this->memory(1));
    auto ws = reinterpret_cast<const uint8_t *>(
            this->input_memory(pd()->ws_idx()));

    auto scratchpad = this->scratchpad();

    bnorm_driver_->exec_bwd(src, diff_src, diff_dst,
            scale_shift, diff_scale_shift, mean, var, ws, scratchpad);

    e->set_state(event_t::ready);
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_bwd_t<isa>::
~jit_uni_tbb_batch_normalization_bwd_t() { delete bnorm_driver_; }

/* struct instantiation */
template struct jit_uni_tbb_batch_normalization_bwd_t<avx2>;
template struct jit_uni_tbb_batch_normalization_bwd_t<avx512_common>;

}
}
}
