defmodule Triton.MLIR.Typespec do
  defstruct [:shape, :type]

  def tensor(type, shape) do
    struct(__MODULE__, shape: shape, type: type)
  end

  def tuple(children) when is_list(children) do
    struct(__MODULE__, shape: children, type: :tuple)
  end

  def type_to_string(%__MODULE__{shape: children, type: :tuple}) when is_list(children) do
    children = Enum.map_join(children, ", ", &to_string/1)
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
  defp type_number({:s, width}), do: "i#{width}"
  defp type_number({:u, width}), do: "ui#{width}"
  defp type_number({:f, 8}), do: "f8E5M2"
  defp type_number({:f, width}), do: "f#{width}"
  defp type_number({:bf, width}), do: "bf#{width}"
  defp type_number({:c, 64}), do: "complex<f32>"
  defp type_number({:c, 128}), do: "complex<f64>"
end