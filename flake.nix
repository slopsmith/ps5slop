{
  description = "ps5slop: PS5 compatibility layer for Apple Silicon";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/0bb2eedc20e4739f29e8045034f2828ad7054f8f";
  inputs.slopsmith-ci.url = "github:slopsmith/ci";
  inputs.slopsmith-ci.inputs.nixpkgs.follows = "nixpkgs";

  outputs =
    { nixpkgs, slopsmith-ci, ... }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
      # Only the libraries ffmpeg-next links, no external codecs, LGPL (docs/audio.md).
      # Its own tests need the libraries left out.
      ffmpeg = (pkgs.ffmpeg-headless.override {
        withHeadlessDeps = false;
        withSmallDeps = false;
        withFullDeps = false;
        withGPL = false;
        withVersion3 = false;
        buildAvcodec = true;
        buildAvformat = true;
        buildAvutil = true;
        buildFfmpeg = false;
        buildFfprobe = false;
        withDocumentation = false;
      }).overrideAttrs { doCheck = false; };
    in
    {
      packages.${system}.slop-app = slopsmith-ci.packages.${system}.slop-app;

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
        buildInputs = [ ffmpeg ];
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      };
    };
}
