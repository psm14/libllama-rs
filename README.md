# libllama-rs

Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Overview

`libllama-rs` is a wrapper around `llama.cpp`'s C API that provides a slightly more idiomatic Rust interface to it.

## Basic Usage

```rust
let mut llama = LLaMaCPP::from_file(&file, LLaMaCPP::default_params()).expect("Failed to load model");

let prompt = "### Instruction:
You are ChatLLaMa, an honest and helpful chatbot

### Input:
How tall is the Eiffel Tower?

### Response:
";

let continuation = llama.iter()
  .consume_bos()
  .consume(&prompt)
  .collect::<String>();

println!("{}{}", prompt, continuation);
```
