// Autogenerated file by gen-gpu-ops.py. Do not edit directly!

#include <ATen/Tensor.h>
#include <ATen/ipex_type_dpcpp_customized.h>

namespace at {

namespace AtenIpexTypeDPCPP {
  at::Tensor & abs_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & acos_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  at::Tensor & add_(at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  at::Tensor & add_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  at::Tensor add(const at::Tensor & self, at::Scalar other, at::Scalar alpha);
  at::Tensor & add_(at::Tensor & self, at::Scalar other, at::Scalar alpha);
  at::Tensor all(const at::Tensor & self, int64_t dim, bool keepdim);
  at::Tensor & all_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim);
  at::Tensor & arange_out(at::Tensor & out, at::Scalar start, at::Scalar end, at::Scalar step);
  at::Tensor as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset);
  at::Tensor & asin_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & atan_(at::Tensor & self);
  at::Tensor & atan_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor baddbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, at::Scalar beta, at::Scalar alpha);
  at::Tensor & baddbmm_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, at::Scalar beta, at::Scalar alpha);
  at::Tensor & baddbmm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, at::Scalar beta, at::Scalar alpha);
  at::Tensor & bernoulli_(at::Tensor & self, const at::Tensor & p, at::Generator * generator);
  at::Tensor & bernoulli_(at::Tensor & self, double p, at::Generator * generator);
  at::Tensor bincount(const at::Tensor & self, const at::Tensor & weights, int64_t minlength);
  at::Tensor bitwise_not(const at::Tensor & self);
  at::Tensor & bitwise_not_(at::Tensor & self);
  at::Tensor & bitwise_not_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor logical_not(const at::Tensor & self);
  at::Tensor & logical_not_(at::Tensor & self);
  at::Tensor & logical_not_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & logical_xor_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor bmm(const at::Tensor & self, const at::Tensor & mat2);
  at::Tensor & bmm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat2);
  at::Tensor & ceil_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & clamp_(at::Tensor & self, c10::optional<at::Scalar> min, c10::optional<at::Scalar> max);
  at::Tensor & clamp_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::Scalar> min, c10::optional<at::Scalar> max);
  at::Tensor & clamp_max_(at::Tensor & self, at::Scalar max);
  at::Tensor & clamp_max_out(at::Tensor & out, const at::Tensor & self, at::Scalar max);
  at::Tensor & clamp_min_(at::Tensor & self, at::Scalar min);
  at::Tensor & clamp_min_out(at::Tensor & out, const at::Tensor & self, at::Scalar min);
  at::Tensor convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
  std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask);
  at::Tensor & copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking);
  at::Tensor & cos_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & cosh_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor div(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & div_(at::Tensor & self, const at::Tensor & other);
  at::Tensor & div_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
  std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const at::Tensor & per_sample_weights);
  at::Tensor _embedding_bag_dense_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const at::Tensor & per_sample_weights);
  at::Tensor empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format);
  at::Tensor & resize_(at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format);
  at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, const at::TensorOptions & options);
  at::Tensor & erf_(at::Tensor & self);
  at::Tensor & erf_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & erfc_(at::Tensor & self);
  at::Tensor & erfc_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & exp_(at::Tensor & self);
  at::Tensor & exp_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & expm1_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & eye_out(at::Tensor & out, int64_t n);
  at::Tensor & eye_out(at::Tensor & out, int64_t n, int64_t m);
  at::Tensor & fill_(at::Tensor & self, at::Scalar value);
  at::Tensor & fill_(at::Tensor & self, const at::Tensor & value);
  at::Tensor & floor_out(at::Tensor & out, const at::Tensor & self);
  std::tuple<at::Tensor &,at::Tensor &> kthvalue_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim);
  std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, int64_t M, int64_t N, double eps);
  std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const at::Tensor & weight, int64_t M, int64_t N, std::array<bool,3> output_mask);
  at::Tensor & linspace_out(at::Tensor & out, at::Scalar start, at::Scalar end, int64_t steps);
  at::Tensor & log_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & log10_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & log1p_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & log2_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & logspace_out(at::Tensor & out, at::Scalar start, at::Scalar end, int64_t steps, double base);
  at::Tensor _log_softmax(const at::Tensor & self, int64_t dim, bool half_to_float);
  at::Tensor _log_softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self);
  std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, int64_t dim, bool keepdim);
  std::tuple<at::Tensor &,at::Tensor &> max_out(at::Tensor & max, at::Tensor & max_values, const at::Tensor & self, int64_t dim, bool keepdim);
  std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, int64_t dim, bool keepdim);
  std::tuple<at::Tensor &,at::Tensor &> min_out(at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, int64_t dim, bool keepdim);
  at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2);
  at::Tensor & mm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat2);
  at::Tensor mul(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & mul_(at::Tensor & self, const at::Tensor & other);
  at::Tensor & mul_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor mul(const at::Tensor & self, at::Scalar other);
  at::Tensor & mul_(at::Tensor & self, at::Scalar other);
  at::Tensor narrow_copy(const at::Tensor & self, int64_t dim, int64_t start, int64_t length);
  std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & running_mean, const at::Tensor & running_var, bool training, double momentum, double eps);
  std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & running_mean, const at::Tensor & running_var, const at::Tensor & save_mean, const at::Tensor & save_invstd, bool train, double eps, std::array<bool,3> output_mask);
  at::Tensor & range_out(at::Tensor & out, at::Scalar start, at::Scalar end, at::Scalar step);
  at::Tensor & neg_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor repeat_interleave(const at::Tensor & repeats);
  at::Tensor & round_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor relu(const at::Tensor & self);
  at::Tensor & relu_(at::Tensor & self);
  at::Tensor hardshrink(const at::Tensor & self, at::Scalar lambd);
  at::Tensor hardshrink_backward(const at::Tensor & grad_out, const at::Tensor & self, at::Scalar lambd);
  at::Tensor & rsqrt_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor sigmoid(const at::Tensor & self);
  at::Tensor & sigmoid_(at::Tensor & self);
  at::Tensor & sigmoid_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & sin_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & sinh_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor _softmax(const at::Tensor & self, int64_t dim, bool half_to_float);
  at::Tensor _softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self);
  at::Tensor sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
  at::Tensor sum(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype);
  at::Tensor & sum_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype);
  at::Tensor & sqrt_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & tan_(at::Tensor & self);
  at::Tensor & tan_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & tanh_(at::Tensor & self);
  at::Tensor & tanh_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor threshold(const at::Tensor & self, at::Scalar threshold, at::Scalar value);
  at::Tensor & threshold_(at::Tensor & self, at::Scalar threshold, at::Scalar value);
  at::Tensor & threshold_out(at::Tensor & out, const at::Tensor & self, at::Scalar threshold, at::Scalar value);
  at::Tensor roll(const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims);
  at::Tensor & trunc_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor _s_where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other);
  at::Tensor norm(const at::Tensor & self, c10::optional<at::Scalar> p, at::ScalarType dtype);
  at::Tensor norm(const at::Tensor & self, at::Scalar p);
  at::Tensor norm(const at::Tensor & self, c10::optional<at::Scalar> p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype);
  at::Tensor norm(const at::Tensor & self, c10::optional<at::Scalar> p, at::IntArrayRef dim, bool keepdim);
  at::Tensor & norm_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::Scalar> p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype);
  at::Tensor & norm_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::Scalar> p, at::IntArrayRef dim, bool keepdim);
  at::Tensor clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format);
  at::Tensor & resize_as_(at::Tensor & self, const at::Tensor & the_template, c10::optional<at::MemoryFormat> memory_format);
  at::Tensor & pow_out(at::Tensor & out, const at::Tensor & self, at::Scalar exponent);
  at::Tensor pow(const at::Tensor & self, at::Scalar exponent);
  at::Tensor & zero_(at::Tensor & self);
  at::Tensor & sub_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  at::Tensor sub(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  at::Tensor & sub_(at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  at::Tensor sub(const at::Tensor & self, at::Scalar other, at::Scalar alpha);
  at::Tensor & sub_(at::Tensor & self, at::Scalar other, at::Scalar alpha);
  at::Tensor rsub(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  at::Tensor rsub(const at::Tensor & self, at::Scalar other, at::Scalar alpha);
  at::Tensor addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha);
  at::Scalar _local_scalar_dense(const at::Tensor & self);
  at::Tensor & set_(at::Tensor & self, at::Storage source);
  at::Tensor & set_(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride);
  at::Tensor & set_(at::Tensor & self, const at::Tensor & source);
  at::Tensor & set_(at::Tensor & self);
  bool is_set_to(const at::Tensor & self, const at::Tensor & tensor);
  at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, at::Scalar value);
  at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value);
  at::Tensor & masked_scatter_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source);
  at::Tensor view(const at::Tensor & self, at::IntArrayRef size);
  at::Tensor & index_add_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source);
  at::Tensor & index_fill_(at::Tensor & self, int64_t dim, const at::Tensor & index, at::Scalar value);
  at::Tensor & index_fill_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value);
  at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src);
  at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, at::Scalar value);
  at::Tensor & scatter_add_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src);
  at::Tensor __and__(const at::Tensor & self, at::Scalar other);
  at::Tensor __and__(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & __iand__(at::Tensor & self, at::Scalar other);
  at::Tensor & __iand__(at::Tensor & self, const at::Tensor & other);
  at::Tensor __or__(const at::Tensor & self, at::Scalar other);
  at::Tensor __or__(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & __ior__(at::Tensor & self, at::Scalar other);
  at::Tensor & __ior__(at::Tensor & self, const at::Tensor & other);
  at::Tensor & tril_(at::Tensor & self, int64_t diagonal);
  at::Tensor & triu_(at::Tensor & self, int64_t diagonal);
  at::Tensor & digamma_(at::Tensor & self);
  at::Tensor & renorm_(at::Tensor & self, at::Scalar p, int64_t dim, at::Scalar maxnorm);
  at::Tensor & pow_(at::Tensor & self, at::Scalar exponent);
  at::Tensor & pow_(at::Tensor & self, const at::Tensor & exponent);
  at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, at::Scalar weight);
  at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight);
  at::Tensor & fmod_(at::Tensor & self, at::Scalar other);
  at::Tensor & fmod_(at::Tensor & self, const at::Tensor & other);
  at::Tensor & remainder_(at::Tensor & self, at::Scalar other);
  at::Tensor & remainder_(at::Tensor & self, const at::Tensor & other);
  at::Tensor & addcdiv_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, at::Scalar value);
  at::Tensor & normal_(at::Tensor & self, double mean, double std, at::Generator * generator);
  at::Tensor & diag_out(at::Tensor & out, const at::Tensor & self, int64_t diagonal);
  at::Tensor diag(const at::Tensor & self, int64_t diagonal);
  at::Tensor & triu_out(at::Tensor & out, const at::Tensor & self, int64_t diagonal);
  at::Tensor & tril_out(at::Tensor & out, const at::Tensor & self, int64_t diagonal);
  at::Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const at::TensorOptions & options);
  at::Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const at::TensorOptions & options);
  at::Tensor trace(const at::Tensor & self);
  at::Tensor & ne_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor ne(const at::Tensor & self, at::Scalar other);
  at::Tensor & ne_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor ne(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & eq_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor eq(const at::Tensor & self, at::Scalar other);
  at::Tensor & eq_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor eq(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & ge_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor ge(const at::Tensor & self, at::Scalar other);
  at::Tensor & ge_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor ge(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & le_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor le(const at::Tensor & self, at::Scalar other);
  at::Tensor & le_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor le(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & gt_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor gt(const at::Tensor & self, at::Scalar other);
  at::Tensor & gt_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor gt(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & lt_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor lt(const at::Tensor & self, at::Scalar other);
  at::Tensor & lt_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor lt(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & index_select_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index);
  at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);
  at::Tensor & masked_select_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mask);
  at::Tensor masked_select(const at::Tensor & self, const at::Tensor & mask);
  at::Tensor & nonzero_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor nonzero(const at::Tensor & self);
  at::Tensor & gather_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad);
  at::Tensor gather(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad);
  at::Tensor & addcmul_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, at::Scalar value);
  at::Tensor addcmul(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, at::Scalar value);
  at::Tensor & addcmul_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, at::Scalar value);
  at::Tensor & addcdiv_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, at::Scalar value);
  at::Tensor addcdiv(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, at::Scalar value);
  at::Tensor & digamma_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor digamma(const at::Tensor & self);
  at::Tensor erfinv(const at::Tensor & self);
  at::Tensor & erfinv_(at::Tensor & self);
  at::Tensor & erfinv_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor sign(const at::Tensor & self);
  at::Tensor & sign_(at::Tensor & self);
  at::Tensor & sign_out(at::Tensor & out, const at::Tensor & self);
  at::Tensor & lerp_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & end, at::Scalar weight);
  at::Tensor & lerp_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight);
  at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, at::Scalar weight);
  at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight);
  at::Tensor & fmod_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor fmod(const at::Tensor & self, at::Scalar other);
  at::Tensor & fmod_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor fmod(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & remainder_out(at::Tensor & out, const at::Tensor & self, at::Scalar other);
  at::Tensor remainder(const at::Tensor & self, at::Scalar other);
  at::Tensor & remainder_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor remainder(const at::Tensor & self, const at::Tensor & other);
  at::Tensor & min_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor min(const at::Tensor & self, const at::Tensor & other);
  at::Tensor min(const at::Tensor & self);
  at::Tensor & max_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  at::Tensor max(const at::Tensor & self, const at::Tensor & other);
  at::Tensor max(const at::Tensor & self);
  at::Tensor median(const at::Tensor & self);
  std::tuple<at::Tensor &,at::Tensor &> sort_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim, bool descending);
  std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, int64_t dim, bool descending);
  std::tuple<at::Tensor &,at::Tensor &> topk_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
  std::tuple<at::Tensor,at::Tensor> topk(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
  at::Tensor all(const at::Tensor & self);
  at::Tensor & renorm_out(at::Tensor & out, const at::Tensor & self, at::Scalar p, int64_t dim, at::Scalar maxnorm);
  at::Tensor renorm(const at::Tensor & self, at::Scalar p, int64_t dim, at::Scalar maxnorm);
  at::Tensor unfold(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step);
  at::Tensor & pow_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & exponent);
  at::Tensor pow(const at::Tensor & self, const at::Tensor & exponent);
  at::Tensor & pow_out(at::Tensor & out, at::Scalar self, const at::Tensor & exponent);
  at::Tensor pow(at::Scalar self, const at::Tensor & exponent);
  at::Tensor _cumsum(const at::Tensor & self, int64_t dim);
  at::Tensor & _cumsum_out(at::Tensor & out, const at::Tensor & self, int64_t dim);
  at::Tensor _cumprod(const at::Tensor & self, int64_t dim);
  at::Tensor & _cumprod_out(at::Tensor & out, const at::Tensor & self, int64_t dim);
  at::Tensor _cat(at::TensorList tensors, int64_t dim);
  at::Tensor & _cat_out(at::Tensor & out, at::TensorList tensors, int64_t dim);
  std::tuple<at::Tensor,at::Tensor> _max(const at::Tensor & self, int64_t dim, bool keepdim);
  std::tuple<at::Tensor &,at::Tensor &> _max_out(at::Tensor & max, at::Tensor & max_indices, const at::Tensor & self, int64_t dim, bool keepdim);
  std::tuple<at::Tensor,at::Tensor> _min(const at::Tensor & self, int64_t dim, bool keepdim);
  std::tuple<at::Tensor &,at::Tensor &> _min_out(at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, int64_t dim, bool keepdim);
  at::Tensor & binary_cross_entropy_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction);
  at::Tensor binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction);
  at::Tensor & binary_cross_entropy_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction);
  at::Tensor binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction);
  at::Tensor & mse_loss_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & target, int64_t reduction);
  at::Tensor mse_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction);
  std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out(at::Tensor & output, at::Tensor & total_weight, const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction, int64_t ignore_index);
  std::tuple<at::Tensor,at::Tensor> nll_loss_forward(const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction, int64_t ignore_index);
  at::Tensor & nll_loss_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
  at::Tensor nll_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Tensor & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
  at::Tensor & hardtanh_out(at::Tensor & out, const at::Tensor & self, at::Scalar min_val, at::Scalar max_val);
  at::Tensor hardtanh(const at::Tensor & self, at::Scalar min_val, at::Scalar max_val);
  at::Tensor & hardtanh_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, at::Scalar min_val, at::Scalar max_val);
  at::Tensor hardtanh_backward(const at::Tensor & grad_output, const at::Tensor & self, at::Scalar min_val, at::Scalar max_val);
  at::Tensor & hardtanh_(at::Tensor & self, at::Scalar min_val, at::Scalar max_val);
  at::Tensor & softplus_out(at::Tensor & out, const at::Tensor & self, at::Scalar beta, at::Scalar threshold);
  at::Tensor softplus(const at::Tensor & self, at::Scalar beta, at::Scalar threshold);
  at::Tensor & softplus_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, at::Scalar beta, at::Scalar threshold, const at::Tensor & output);
  at::Tensor softplus_backward(const at::Tensor & grad_output, const at::Tensor & self, at::Scalar beta, at::Scalar threshold, const at::Tensor & output);
  at::Tensor & adaptive_avg_pool2d_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef output_size);
  at::Tensor adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size);
  at::Tensor _adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size);
  at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self);
  at::Tensor & avg_pool2d_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  at::Tensor avg_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  at::Tensor & avg_pool2d_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  at::Tensor avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  at::Tensor & avg_pool3d_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  at::Tensor avg_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  at::Tensor & avg_pool3d_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  at::Tensor avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
  std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out(at::Tensor & out, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
  std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
  at::Tensor & max_pool2d_with_indices_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices);
  at::Tensor max_pool2d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices);
  at::Tensor & upsample_nearest2d_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef output_size);
  at::Tensor upsample_nearest2d(const at::Tensor & self, at::IntArrayRef output_size);
  at::Tensor & upsample_nearest2d_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size);
  at::Tensor upsample_nearest2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size);
  at::Tensor & sigmoid_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & output);
  at::Tensor sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & output);
  at::Tensor & tanh_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & output);
  at::Tensor tanh_backward(const at::Tensor & grad_output, const at::Tensor & output);


} // namespace AtenIpexTypeDPCPP
} // namespace at

