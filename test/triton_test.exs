defmodule TritonTest do
  use ExUnit.Case

  alias Triton.Language, as: Tl

  describe "jit" do
    test "jits a function" do
      fun = fn x ->
        Tl.abs(x)
      end

      Triton.jit(fun) |> IO.inspect
    end
  end
end
