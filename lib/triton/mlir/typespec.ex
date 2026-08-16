defmodule Triton.MLIR.Typespec do
  defstruct [:shape, :type]

  @type t :: %__MODULE__{shape: tuple() | list(), type: term()}

  @dtype_aliases %{
    bool: {:pred, 8},
    int1: {:pred, 8},
    s8: {:s, 8},
    s16: {:s, 16},
    s32: {:s, 32},
    s64: {:s, 64},
    u8: {:u, 8},
    u16: {:u, 16},
    u32: {:u, 32},
    u64: {:u, 64},
    f8: {:f, 8},
    f16: {:f, 16},
    f32: {:f, 32},
    f64: {:f, 64},
    int8: {:s, 8},
    int16: {:s, 16},
    int32: {:s, 32},
    int64: {:s, 64},
    uint8: {:u, 8},
    uint16: {:u, 16},
    uint32: {:u, 32},
    uint64: {:u, 64},
    float8: {:f, 8},
    float16: {:f, 16},
    float32: {:f, 32},
    float64: {:f, 64},
    fp8: {:f, 8},
    fp16: {:f, 16},
    fp32: {:f, 32},
    fp64: {:f, 64},
    bfloat16: {:bf, 16},
    bf16: {:bf, 16},
    complex64: {:c, 64},
    complex128: {:c, 128}
  }

  def tensor(type, shape) do
    type = normalize_type(type)
    validate_type!(type)
    struct(__MODULE__, shape: normalize_shape!(shape, type), type: type)
  end

  def scalar(type), do: tensor(type, {})

  def pointer(type) do
    type = normalize_type(type)
    validate_element_type!(type)
    {:ptr, type}
  end

  def ptr(type), do: pointer(type)

  def from(%__MODULE__{} = spec), do: spec

  def from(nil), do: struct(__MODULE__, shape: nil, type: :void)

  def from(%Nx.Tensor{shape: shape, type: type}) when is_tuple(shape) do
    tensor(type, shape)
  end

  def from(%{shape: shape, type: type, dtype: dtype})
      when is_integer(shape) or is_tuple(shape) or is_list(shape) do
    type = normalize_type(type)
    dtype = normalize_type(dtype)

    if type == dtype do
      tensor(type, shape)
    else
      raise ArgumentError,
            "Triton argument spec type and dtype metadata cannot both be set to different values"
    end
  end

  def from(%{shape: shape, type: type})
      when is_integer(shape) or is_tuple(shape) or is_list(shape) do
    tensor(type, shape)
  end

  def from(%{shape: shape, dtype: dtype})
      when is_integer(shape) or is_tuple(shape) or is_list(shape) do
    tensor(dtype, shape)
  end

  def from(%{shape: shape} = value)
      when is_integer(shape) or is_tuple(shape) or is_list(shape) do
    values =
      value
      |> tensor_map_values!()
      |> flatten_values!()

    normalized_shape = normalize_shape!(shape)
    validate_value_count!(normalized_shape, values, value)
    tensor(infer_type(values), normalized_shape)
  end

  def from([_head | _tail] = values) do
    {shape, flattened} = flatten_shape!(values)
    tensor(infer_type(flattened), shape)
  end

  def from([]) do
    raise ArgumentError, "cannot infer a Triton argument spec from an empty list"
  end

  def from(value) when is_integer(value), do: scalar({:s, 64})
  def from(value) when is_float(value), do: scalar({:f, 64})
  def from(value) when is_boolean(value), do: scalar({:pred, 8})
  def from({:ptr, _type} = pointer_type), do: scalar(pointer_type)
  def from(value) when is_tuple(value), do: value |> Tuple.to_list() |> tuple()

  def from(other) do
    raise ArgumentError,
          "cannot infer a Triton argument spec from #{inspect(other)}; pass a Triton.MLIR.Typespec or an Nx tensor"
  end

  def tuple(children) when is_list(children) do
    children = Enum.map(children, &from/1)
    struct(__MODULE__, shape: children, type: :tuple)
  end

  defp normalize_shape!(shape, :tuple), do: shape
  defp normalize_shape!(shape, _type), do: normalize_shape!(shape)

  defp normalize_shape!(nil), do: nil

  defp normalize_shape!(shape) when is_tuple(shape) do
    validate_shape!(shape)
    shape
  end

  defp normalize_shape!(shape) when is_list(shape) do
    shape = List.to_tuple(shape)
    validate_shape!(shape)
    shape
  end

  defp normalize_shape!(shape) when is_integer(shape), do: normalize_shape!({shape})

  defp normalize_shape!(shape) do
    raise ArgumentError,
          "Triton tensor spec shape must be an integer, tuple, or list of non-negative integers, got #{inspect(shape)}"
  end

  defp validate_shape!(shape) do
    unless shape |> Tuple.to_list() |> Enum.all?(&(is_integer(&1) and &1 >= 0)) do
      raise ArgumentError,
            "Triton tensor spec shape must be an integer, tuple, or list of non-negative integers, got #{inspect(shape)}"
    end
  end

  def normalize_type({:ptr, type}), do: {:ptr, normalize_type(type)}
  def normalize_type(type) when is_atom(type), do: Map.get(@dtype_aliases, type, type)
  def normalize_type(type), do: type

  def validate_element_type!(type) do
    unless element_type?(type) do
      raise ArgumentError, "Triton element type #{inspect(type)} is not supported"
    end

    :ok
  end

  def validate_type!(type) when type in [:unknown, :void, :tuple], do: :ok
  def validate_type!(type), do: validate_element_type!(type)

  def element_type?({:ptr, type}), do: element_type?(type)
  def element_type?({:pred, 8}), do: true
  def element_type?({kind, width}) when kind in [:s, :u] and width in [8, 16, 32, 64], do: true
  def element_type?({:f, width}) when width in [8, 16, 32, 64], do: true
  def element_type?({:bf, 16}), do: true
  def element_type?({:c, width}) when width in [64, 128], do: true
  def element_type?(_type), do: false

  def type_to_string(%__MODULE__{type: :void}), do: "void"

  def type_to_string(%__MODULE__{shape: nil, type: type}) do
    raise ArgumentError, "Triton tensor spec shape is missing for type #{inspect(type)}"
  end

  def type_to_string(%__MODULE__{shape: children, type: :tuple}) when is_list(children) do
    children = Enum.map_join(children, ", ", &type_to_string/1)
    "tuple<#{children}>"
  end

  def type_to_string(%__MODULE__{shape: shape, type: type}) do
    shape_sequence = shape |> Tuple.to_list() |> Enum.map_join("", &"#{&1}x")
    "tensor<#{shape_sequence}#{type_number(type)}>"
  end

  def encode(mod) do
    mod
    |> type_to_string()
    |> to_charlist()
  end

  defp type_number({:pred, 8}), do: "i1"
  defp type_number({:ptr, type}), do: "!tt.ptr<#{type_number(type)}>"
  defp type_number(:unknown), do: "?"
  defp type_number(:void), do: "void"
  defp type_number({:s, width}), do: "i#{width}"
  defp type_number({:u, width}), do: "ui#{width}"
  defp type_number({:f, 8}), do: "f8E5M2"
  defp type_number({:f, width}), do: "f#{width}"
  defp type_number({:bf, width}), do: "bf#{width}"
  defp type_number({:c, 64}), do: "complex<f32>"
  defp type_number({:c, 128}), do: "complex<f64>"

  defp type_number(type) do
    raise ArgumentError, "Triton element type #{inspect(type)} is not supported"
  end

  defp flatten_shape!(values) when is_list(values) do
    cond do
      values == [] ->
        {{0}, []}

      Enum.all?(values, &is_list/1) ->
        child_shapes_and_values = Enum.map(values, &flatten_shape!/1)
        [{child_shape, _} | _] = child_shapes_and_values

        unless Enum.all?(child_shapes_and_values, &(elem(&1, 0) == child_shape)) do
          raise ArgumentError,
                "cannot infer a rectangular Triton tensor shape from #{inspect(values)}"
        end

        flattened = Enum.flat_map(child_shapes_and_values, &elem(&1, 1))
        shape = [length(values) | Tuple.to_list(child_shape)] |> List.to_tuple()
        {shape, flattened}

      Enum.any?(values, &is_list/1) ->
        raise ArgumentError,
              "cannot infer a rectangular Triton tensor shape from #{inspect(values)}"

      true ->
        {{length(values)}, values}
    end
  end

  defp tensor_map_values!(value) do
    cond do
      Map.has_key?(value, :values) ->
        Map.fetch!(value, :values)

      Map.has_key?(value, :data) ->
        Map.fetch!(value, :data)

      Map.has_key?(value, :value) ->
        Map.fetch!(value, :value)

      true ->
        raise ArgumentError,
              "cannot infer a Triton argument spec from tensor-like map #{inspect(value)} without :type, :dtype, :values, :data, or :value metadata"
    end
  end

  defp flatten_values!(values) when is_list(values) do
    {_shape, flattened} = flatten_shape!(values)
    flattened
  end

  defp flatten_values!(value), do: [value]

  defp validate_value_count!(shape, values, source) do
    expected = shape |> Tuple.to_list() |> Enum.product()

    unless length(values) == expected do
      raise ArgumentError,
            "Triton argument spec for shape #{inspect(shape)} must contain #{expected} values, got #{length(values)} in #{inspect(source)}"
    end
  end

  defp infer_type([]) do
    raise ArgumentError, "cannot infer a Triton element type from []"
  end

  defp infer_type(values) do
    cond do
      Enum.any?(values, &is_float/1) -> {:f, 64}
      Enum.all?(values, &is_boolean/1) -> {:pred, 8}
      Enum.all?(values, &is_integer/1) -> {:s, 64}
      true -> raise ArgumentError, "cannot infer a Triton element type from #{inspect(values)}"
    end
  end
end
