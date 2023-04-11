extern crate bindgen;

use std::env;
use std::path::PathBuf;

use bindgen::CargoCallbacks;

fn main() {
    let libdir_path = PathBuf::from("libllama")
      .canonicalize()
      .expect("cannot canonicalize path");

    let headers_path = libdir_path.join("llama.h");
    let headers_path_str = headers_path.to_str().expect("Path is not a valid string");

    let llama_o_path = libdir_path.join("llama.o");
    let ggml_o_path = libdir_path.join("ggml.o");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let llama_a_path = out_path.join("libllama.a");

    println!("cargo:rustc-link-search={}", out_path.to_str().expect("Path is not a valid string"));
    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rerun-if-changed={}", headers_path_str);

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    cxx_build::bridge("src/lib.rs").flag_if_supported("-std=c++11");

    if !std::process::Command::new("make")
        .current_dir(&libdir_path)
        .arg("llama.o")
        .arg("ggml.o")
        .output()
        .expect("could not spawn `make`")
        .status
        .success()
    {
        panic!("could not compile llama.cpp");
    }

    if !std::process::Command::new("ar")
        .arg("rcs")
        .arg(llama_a_path.to_str().expect("Path is not a valid string"))
        .arg(llama_o_path.to_str().expect("Path is not a valid string"))
        .arg(ggml_o_path.to_str().expect("Path is not a valid string"))
        .output()
        .expect("could not move shared object file")
        .status
        .success()
    {
        panic!("could not create static library");
    }

    let bindings = bindgen::Builder::default()
      .header(headers_path_str)
      .parse_callbacks(Box::new(CargoCallbacks))
      .generate()
      .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
