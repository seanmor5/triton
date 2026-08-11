#ifndef TRITON_CUDA_H_
#define TRITON_CUDA_H_

// CUDA driver runtime for the Triton Elixir NIF.
//
// The CUDA driver library is loaded lazily with dlopen so the NIF builds and
// loads on machines without any CUDA installation. Every entry point reports
// a structured {:error, reason} when the driver is unavailable.

#include "erl_nif.h"

int triton_cuda_open_resources(ErlNifEnv* env);

ERL_NIF_TERM cuda_available(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_driver_version(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_device_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_device_info(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_load_executable(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_launch(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_mem_alloc(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_mem_free(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_memcpy_htod(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_memcpy_dtoh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_memset_d8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM cuda_synchronize(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

#endif
