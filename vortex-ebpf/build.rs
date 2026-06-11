// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(feature = "host"))]
fn main() {}

#[cfg(feature = "host")]
fn main() -> anyhow::Result<()> {
    build::main()
}

#[cfg(feature = "host")]
mod build {
    //! Compiles the kernel-side `vortex-ebpf-ebpf` crate for the BPF target via
    //! `aya-build` and places the object in `OUT_DIR` for the host loader to embed.
    //!
    //! Set `VORTEX_EBPF_SKIP_BPF=1` to skip the kernel build (e.g. CI hosts
    //! without `bpf-linker`); an empty stub is embedded so the host code still
    //! compiles, but the loader will not function.

    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use anyhow::Context;
    use anyhow::anyhow;
    use aya_build::Package;
    use aya_build::Toolchain;

    const EBPF_PACKAGE: &str = "vortex-ebpf-ebpf";
    const EBPF_BIN: &str = "vortex-ebpf-kern";
    const EBPF_TOOLCHAIN: &str = "nightly-2026-03-22";

    const SETUP_HELP: &str = "\
building the Vortex eBPF kernel programs failed.\n\n\
This is a one-time setup (the kernel object is compiled from vortex-ebpf/ebpf at build time, never committed).\n\n\
The nightly toolchain and rust-src are declared in vortex-ebpf/ebpf/rust-toolchain.toml and are installed automatically by rustup on first use. The only manual step is the linker:\n\n    cargo install bpf-linker\n\n\
Or set VORTEX_EBPF_SKIP_BPF=1 to embed a non-functional stub (the crate compiles; profiling is disabled).";

    pub fn main() -> anyhow::Result<()> {
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").context("OUT_DIR not set")?);
        let object = out_dir.join(EBPF_BIN);

        println!("cargo:rerun-if-env-changed=VORTEX_EBPF_SKIP_BPF");
        println!("cargo:rerun-if-changed=src/types.rs");
        println!("cargo:rerun-if-changed=ebpf/src/main.rs");
        println!("cargo:rerun-if-changed=ebpf/Cargo.toml");

        let skip = env::var("VORTEX_EBPF_SKIP_BPF")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if skip {
            fs::write(&object, [])?;
            println!(
                "cargo:warning=VORTEX_EBPF_SKIP_BPF set: embedding empty eBPF stub; the loader will not function"
            );
            return Ok(());
        }

        const ROOT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/ebpf");
        env::set_current_dir(ROOT_DIR).context("entering vortex-ebpf/ebpf")?;

        aya_build::build_ebpf(
            [Package {
                name: EBPF_PACKAGE,
                root_dir: ROOT_DIR,
                no_default_features: false,
                features: &[],
            }],
            Toolchain::Custom(EBPF_TOOLCHAIN),
        )
        .map_err(|e| anyhow!("{e:#}\n\n{SETUP_HELP}"))
    }
}
