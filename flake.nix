{
  description = "ps5slop: PS5 compatibility layer for Apple Silicon";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/0bb2eedc20e4739f29e8045034f2828ad7054f8f";

  outputs =
    { nixpkgs, ... }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = [
          pkgs.cargo
          pkgs.clippy
          pkgs.rustc
          pkgs.rustfmt
          pkgs.rust-analyzer
          pkgs.lld
          pkgs.nasm
          pkgs.llvm
          pkgs.pkg-config
        ];
        buildInputs = [ pkgs.ffmpeg-headless ];
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      };
    };
}
