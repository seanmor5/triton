#include "triton_nif_util.h"

namespace nif {

ERL_NIF_TERM error(ErlNifEnv* env, const char* msg) {
  ERL_NIF_TERM atom = enif_make_atom(env, "error");
  ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
  return enif_make_tuple2(env, atom, msg_term);
}

ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term) {
  return enif_make_tuple2(env, ok(env), term);
}

ERL_NIF_TERM ok(ErlNifEnv* env) {
  return enif_make_atom(env, "ok");
}


int get(ErlNifEnv* env, ERL_NIF_TERM term, bool* var) {
  int value;
  if (!enif_get_int(env, term, &value)) return 0;
  *var = static_cast<bool>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, int* var) {
  return enif_get_int(env, term, reinterpret_cast<int*>(var));
}

}
