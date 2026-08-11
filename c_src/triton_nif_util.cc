#include "triton_nif_util.h"

#include <cstring>

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

int get(ErlNifEnv* env, ERL_NIF_TERM term, int* var) {
  return enif_get_int(env, term, reinterpret_cast<int*>(var));
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, bool* var) {
  int value;
  if (enif_get_int(env, term, &value)) {
    *var = static_cast<bool>(value);
    return 1;
  }

  char atom[8];
  if (enif_get_atom(env, term, atom, sizeof(atom), ERL_NIF_LATIN1)) {
    if (std::string(atom) == "true") {
      *var = true;
      return 1;
    }
    if (std::string(atom) == "false") {
      *var = false;
      return 1;
    }
  }
  return 0;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string& var) {
  unsigned len;
  int ret = enif_get_list_length(env, term, &len);

  if (!ret) {
    ErlNifBinary bin;
    ret = enif_inspect_binary(env, term, &bin);
    if (!ret) {
      return 0;
    }
    var = std::string((const char*)bin.data, bin.size);
    return ret;
  }

  var.resize(len + 1);
  ret = enif_get_string(env, term, &*(var.begin()), var.size(), ERL_NIF_LATIN1);

  if (ret > 0) {
    var.resize(ret - 1);
  } else if (ret == 0) {
    var.resize(0);
  } else {
  }

  return ret;
}

int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<std::string>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) {
    return 0;
  }
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    std::string elem;
    if (!get(env, head, elem)) {
      return 0;
    }
    var.push_back(elem);
    list = tail;
  }
  return 1;
}

ERL_NIF_TERM make(ErlNifEnv* env, std::string var) {
  return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM make_binary(ErlNifEnv* env, const std::string& var) {
  ERL_NIF_TERM term;
  unsigned char* data = enif_make_new_binary(env, var.size(), &term);
  std::memcpy(data, var.data(), var.size());
  return term;
}

int get_keyword(ErlNifEnv* env, ERL_NIF_TERM list, const char* key,
                ERL_NIF_TERM* value_out) {
  ERL_NIF_TERM head, tail = list;
  ERL_NIF_TERM key_atom = enif_make_atom(env, key);

  while (enif_get_list_cell(env, tail, &head, &tail)) {
    int arity;
    const ERL_NIF_TERM* pair;
    if (!enif_get_tuple(env, head, &arity, &pair) || arity != 2) {
      continue;
    }
    if (enif_compare(pair[0], key_atom) == 0) {
      *value_out = pair[1];
      return 1;
    }
  }
  return 0;
}

}
