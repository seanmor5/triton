defmodule Triton.Language do
  alias Triton.Language.Expr

  import Bitwise

  @triton_max_tensor_numel 1_048_576

  # Programs

  def program_id(axis) when axis in [0, 1, 2] do
    Expr.new(:program_id, [], axis: axis)
  end

  def num_programs(axis) when axis in [0, 1, 2] do
    Expr.new(:num_programs, [], axis: axis)
  end

  # Creation

  def arange(low, high) do
    unless high > low do
      raise "high must be greater than low"
    end

    unless high - low <= @triton_max_tensor_numel do
      raise "number of elements must be less than or equal to #{@triton_max_tensor_numel}"
    end

    unless power_of_two?(low) and power_of_two?(high) do
      raise "both low and high must be powers of two"
    end

    Expr.new(:arange, [], low: low, high: high)
  end

  def cat(%Expr{} = input, %Expr{} = other, opts \\ []) do
    opts = Keyword.validate!(opts, reorder: false)
    Expr.new(:cat, [input, other], opts)
  end

  def full(shape, value, dtype) when is_tuple(shape) and is_integer(value) do
    Expr.new(:full, [], shape: shape, value: value, dtype: dtype)
  end

  def zeros(shape, dtype) when is_tuple(shape) do
    Expr.new(:zeros, [], shape: shape, dtype: dtype)
  end

  def zeros_like(%Expr{} = input) do
    Expr.new(:zeros_like, [input])
  end

  def cast(%Expr{} = input, dtype, opts \\ []) do
    opts = Keyword.validate!(opts, fp_downcast_rounding: :rtne, bitcast: false)
    Expr.new(:cast, [input], [{:dtype, dtype} | opts])
  end

  def broadcast(%Expr{} = input, %Expr{} = other) do
    Expr.new(:broadcast, [input, other])
  end

  def broadcast_to(%Expr{} = input, shape) when is_tuple(shape) do
    Expr.new(:broadcast_to, [input], shape: shape)
  end

  def expand_dims(%Expr{} = input, axis_or_axes)
      when is_integer(axis_or_axes) or is_list(axis_or_axes) do
    axes = List.wrap(axis_or_axes)
    Expr.new(:expand_dims, [input], axes: axes)
  end

  def interleave(%Expr{} = a, %Expr{} = b) do
    Expr.new(:interleave, [a, b])
  end

  def join(%Expr{} = a, %Expr{} = b) do
    Expr.new(:join, [a, b])
  end

  def permute(%Expr{} = input, axes) when is_list(axes) do
    Expr.new(:permute, [input], axes: axes)
  end

  def ravel(%Expr{} = x) do
    Expr.new(:ravel, [x])
  end

  def reshape(%Expr{} = input, shape, opts \\ []) when is_tuple(shape) do
    opts = Keyword.validate!(opts, can_reorder: false)
    Expr.new(:reshape, [input], [{:shape, shape} | opts])
  end

  # TODO:
  # def split(%Expr{} = input) do
  # end

  # TODO:
  # def trans(%Expr{} = input) do
  # end

  def view(%Expr{} = input, shape) when is_tuple(shape) do
    Expr.new(:view, [input], shape: shape)
  end

  def dot(%Expr{} = input, %Expr{} = other) do
    dot(input, other, [])
  end

  def dot(%Expr{} = input, %Expr{} = other, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, input_precision: :tf32)
    Expr.new(:dot, [input, other], opts)
  end

  def dot(%Expr{} = input, %Expr{} = other, %Expr{} = acc) do
    dot(input, other, acc, [])
  end

  def dot(%Expr{} = input, %Expr{} = other, %Expr{} = acc, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, input_precision: :tf32)
    Expr.new(:dot, [input, other, acc], opts)
  end

  # TODO:
  # def load
  # def store
  # def make_block_ptr
  # def advance

  def flip(%Expr{} = x) do
    Expr.new(:flip, [x])
  end

  def where(%Expr{} = condition, %Expr{} = x, %Expr{} = y) do
    Expr.new(:where, [condition, x, y])
  end

  def swizzle_2d(%Expr{} = i, %Expr{} = j, size_i, size_j, size_g)
      when is_integer(size_i) and is_integer(size_j) and is_integer(size_g) do
    Expr.new(:swizzle_2d, [i, j], size_i: size_i, size_j: size_j, size_g: size_g)
  end

  def clamp(%Expr{} = x, %Expr{} = min, %Expr{} = max, opts \\ []) do
    opts = Keyword.validate!(opts, propagate_nan: nil)
    Expr.new(:clamp, [x, min, max], opts)
  end

  def fdiv(%Expr{} = x, %Expr{} = y, opts \\ []) do
    opts = Keyword.validate!(opts, ieee_rounding: false)
    Expr.new(:fdiv, [x, y], opts)
  end

  def fma(%Expr{} = x, %Expr{} = y, %Expr{} = z) do
    Expr.new(:fma, [x, y, z])
  end

  def softmax(%Expr{} = x, opts \\ []) do
    opts = Keyword.validate!(opts, ieee_rounding: false)
    Expr.new(:fdiv, [x], opts)
  end

  @unary_ops [:abs, :ceil, :cos, :erf, :exp, :exp2, :floor] ++
               [:log, :log2, :rsqrt, :sigmoid, :sin, :sqrt, :sqrt_rn]

  for op <- @unary_ops do
    def unquote(op)(%Expr{} = x) do
      Expr.new(unquote(op), [x])
    end
  end

  @binary_ops [:cdiv, :div_rn, :umulhi]

  for op <- @binary_ops do
    def unquote(op)(%Expr{} = x, %Expr{} = y) do
      Expr.new(unquote(op), [x, y])
    end
  end

  for op <- [:maximum, :minimum] do
    def unquote(op)(%Expr{} = x, %Expr{} = y, opts \\ []) do
      opts = Keyword.validate!(opts, propagate_nan: nil)
      Expr.new(unquote(op), [x, y], opts)
    end
  end

  for op <- [:argmax, :argmin] do
    def unquote(op)(%Expr{} = x, axis, opts \\ []) when is_integer(axis) do
      opts = Keyword.validate!(opts, tie_break_left: true, keep_dims: false)
      Expr.new(unquote(op), [x], [{:axis, axis} | opts])
    end
  end

  for op <- [:max, :min] do
    def unquote(op)(%Expr{} = x, opts \\ []) do
      opts =
        Keyword.validate!(opts, [
          :axis,
          keep_dims: false,
          return_indices: false,
          return_indices_tie_break_left: true
        ])

      Expr.new(unquote(op), [x], opts)
    end
  end

  def reduce(%Expr{} = input, fun, opts \\ []) when is_function(fun, 2) do
    opts = Keyword.validate!(opts, [:axis, keep_dims: false])
    Expr.new(:reduce, [input], [{:fun, fun} | opts])
  end

  def sum(%Expr{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:axis, keep_dims: false])
    Expr.new(:sum, [input], opts)
  end

  def xor_sum(%Expr{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:axis, keep_dims: false])
    Expr.new(:xor_sum, [input], opts)
  end

  def associative_scan(%Expr{} = input, axis, fun, opts \\ [])
      when is_integer(axis) and is_function(fun, 2) do
    opts = Keyword.validate!(opts, reverse: false)
    Expr.new(:reduce, [input], [{:fun, fun} | opts])
  end

  def cumprod(%Expr{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, axis: 0, reverse: false)
    Expr.new(:cumprod, [input], opts)
  end

  def cumsum(%Expr{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, axis: 0, reverse: false)
    Expr.new(:cumprod, [input], opts)
  end

  def histogram(%Expr{} = input, num_bins) when is_integer(num_bins) do
    Expr.new(:histogram, [input], num_bins: num_bins)
  end

  def sort(%Expr{} = input, opts \\ []) do
    opts = Keyword.validate!(opts, [:dim, descending: false])
    Expr.new(:sort, [input], opts)
  end

  # TODO: Atomics, Randoms, Iterators, Assembly, Compiler, Debug

  defp power_of_two?(x) do
    (x &&& x - 1) == 0
  end
end
