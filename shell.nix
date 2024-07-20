{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  name = "tinygrad-nix";

  buildInputs = [
    pkgs.clang
  ];

  shellHook = ''
    uv venv
    source .venv/bin/activate
    uv sync
    export PYTHONPATH=${builtins.getEnv "PWD"}:$PYTHONPATH
  '';
}

