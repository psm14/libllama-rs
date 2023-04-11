extern crate libc;
extern crate num_cpus;

// Dummy module just to let `cxx` link against the C++ standard library.
// TODO: What are the magic words to put in `build.rs` to make this unnecessary?
#[cxx::bridge]
mod LLaMaMod {}

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CString;
use std::iter::Iterator;
use std::os::raw::{c_char, c_int};
use std::str;
use std::slice;

#[derive(Debug)]
pub struct LLaMaError(&'static str);

pub struct Params {
  pub n_ctx: c_int,
  pub n_parts: c_int,
  pub seed: c_int,
  pub threads: c_int,
  pub f16_kv: bool,
  pub logits_all: bool,
  pub use_mmap: bool,
  pub use_mlock: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerParams {
  top_p: f32,
  top_k: c_int,
  temperature: f32,
  repeat_penalty: f32,
}

impl SamplerParams {
  pub fn default() -> SamplerParams {
    SamplerParams {
      top_p: 0.9,
      top_k: 40,
      repeat_penalty: 1.17,
      temperature: 1.17,
    }
  }
}

pub struct LLaMaCPP {
  context: *mut llama_context,
  params: Params,
}



impl LLaMaCPP {
  pub fn from_file(path: &str, params: Params) -> Result<LLaMaCPP, LLaMaError> {
    let llama_context_params = llama_context_params {
      n_ctx: params.n_ctx,
      n_parts: params.n_parts,
      seed: params.seed,
      f16_kv: params.f16_kv,
      logits_all: params.logits_all,
      use_mmap: params.use_mmap,
      use_mlock: params.use_mlock,
      embedding: false,
      vocab_only: false,
      progress_callback: None,
      progress_callback_user_data: std::ptr::null_mut(),
    };

    let c_path = CString::new(path).map_err(|_| LLaMaError("Invalid path"))?;
    unsafe {
      // TODO: Does this return a null ptr if the file doesn't exist?
      let context = llama_init_from_file(c_path.as_ptr(), llama_context_params);
      if context == std::ptr::null_mut() {
        Err(LLaMaError("Failed to load model"))
      } else {
        Ok(LLaMaCPP { context, params })
      }
    }
  }

  // Lifetime of the returned string is the same as the LLaMa context
  fn token_to_str(&self, token: llama_token) -> &str {
    unsafe {
      let cstr: *const c_char = llama_token_to_str(self.context, token);
      str::from_utf8_unchecked(slice::from_raw_parts(cstr as *const u8, libc::strlen(cstr)))
    }
  }

  fn tokenize_internal(&self, text: &str) -> Vec<llama_token> {
    let c_text = CString::new(text).unwrap();
    let mut tokens = Vec::with_capacity(self.params.n_ctx as usize);
    unsafe {
      let n_tokens = llama_tokenize(self.context, c_text.as_ptr(), tokens.as_mut_ptr(), self.params.n_ctx, false);
      tokens.set_len(n_tokens as usize);
    }
    tokens
  }

  fn consume_tokens(&mut self, tokens: &[llama_token], n_past: i32) {
    unsafe {
      llama_eval(self.context, tokens.as_ptr(), tokens.len() as i32, n_past, self.params.threads);
    }
  }

  fn sample_token(&self, sampler_params: SamplerParams) -> llama_token {
    unsafe {
      llama_sample_top_p_top_k(self.context, std::ptr::null(), 0, sampler_params.top_k, sampler_params.top_p, sampler_params.temperature, sampler_params.repeat_penalty)
    }
  }

  fn get_logits(&self, last_n_tokens: usize) -> Vec<&[f32]> {
    let mut result = Vec::with_capacity(last_n_tokens);
    unsafe {
      let logits_start = llama_get_logits(self.context) as usize;
      for i in 0..last_n_tokens {
        let row = slice::from_raw_parts((logits_start + i * self.n_vocab() as usize) as *const f32, self.n_vocab() as usize);
        result.push(row);
      }
    }
    result
  }

  fn n_vocab(&self) -> i32 {
    unsafe { llama_n_vocab(self.context) }
  }

  fn n_ctx(&self) -> i32 {
    unsafe { llama_n_ctx(self.context) }
  }

  fn n_embd(&self) -> i32 {
    unsafe { llama_n_embd(self.context) }
  }

  fn threads(&self) -> i32 {
    self.params.threads
  }

  fn system_info() -> &'static str {
    unsafe {
      // This is &'static
      let cstr: *const c_char = llama_print_system_info();
      str::from_utf8_unchecked(slice::from_raw_parts(cstr as *const u8, libc::strlen(cstr)+1))
    }
  }

  fn mmap_supported() -> bool {
    unsafe { llama_mmap_supported() }
  }

  fn mlock_supported() -> bool {
    unsafe { llama_mlock_supported() }
  }

  fn bos_token() -> llama_token {
    unsafe { llama_token_bos() }
  }

  fn eos_token() -> llama_token {
    unsafe { llama_token_eos() }
  }

  pub fn default_params() -> Params {
    let ffi_params = unsafe { llama_context_default_params() };
    Params {
      n_ctx: ffi_params.n_ctx,
      n_parts: ffi_params.n_parts,
      seed: ffi_params.seed,
      threads: num_cpus::get() as i32,
      f16_kv: ffi_params.f16_kv,
      logits_all: ffi_params.logits_all,
      use_mmap: ffi_params.use_mmap,
      use_mlock: ffi_params.use_mlock,
    }
  }
}

impl Drop for LLaMaCPP {
  fn drop(&mut self) {
    unsafe {
      llama_free(self.context);
    }
  }
}

pub trait Tokenizer {
  type Token: Copy;

  fn tokenize(&self, text: &str) -> Vec<Self::Token>;
  fn to_str(&self, token: Self::Token) -> &str;
  fn bos(&self) -> Self::Token;
  fn eos(&self) -> Self::Token;
}

pub trait LLaMa {
  type Token;
  type Tokenizer: Tokenizer<Token = Self::Token>;
  type TokenIterator<'a>: Iterator where Self: 'a;

  fn tokenizer(&self) -> Self::Tokenizer;
  fn token_iter(&mut self, params: SamplerParams) -> Self::TokenIterator<'_>;
}

pub struct LLaMaCPPTokenIter<'a> {
  context: &'a mut LLaMaCPP,
  params: SamplerParams,
  n_past: i32,
  n_last: i32,
}

impl LLaMaCPPTokenIter<'_> {
  fn consume_internal(&mut self, tokens: &[llama_token]) {
    self.context.consume_tokens(tokens, self.n_past);
    self.n_last = tokens.len() as i32;
    self.n_past += self.n_last;
  }

  pub fn consume(mut self, prompt: &str) -> Self {
    let tokens = self.context.tokenize_internal(prompt);
    self.consume_tokens(&tokens)
  }

  pub fn consume_bos(mut self) -> Self {
    self.consume_tokens(&[LLaMaCPP::bos_token()])
  }

  pub fn consume_tokens(mut self, tokens: &[llama_token]) -> Self {
    self.consume_internal(tokens);
    self
  }

  pub fn sample(&self) -> llama_token {
    self.context.sample_token(self.params)
  }

  pub fn get_logits(&self) -> Vec<&[f32]> {
    self.context.get_logits(self.n_last as usize)
  }
}

impl Iterator for LLaMaCPPTokenIter<'_> {
  type Item = String;

  fn next(&mut self) -> Option<Self::Item> {
    let token = self.sample();
    if token == LLaMaCPP::eos_token() {
      return None;
    }
    self.consume_internal(&[token]);
    let token = self.context.token_to_str(token);
    // TODO: Avoid this allocation if possible
    Some(token.to_string())
  }
}

impl Tokenizer for *const LLaMaCPP {
  type Token = llama_token;

  fn tokenize(&self, text: &str) -> Vec<Self::Token> {
    let llama = unsafe { &**self };
    llama.tokenize_internal(text)
  }

  fn to_str(&self, token: Self::Token) -> &str {
    let llama = unsafe { &**self };
    llama.token_to_str(token)
  }

  fn bos(&self) -> Self::Token {
    LLaMaCPP::bos_token()
  }

  fn eos(&self) -> Self::Token {
    LLaMaCPP::eos_token()
  }
}

impl LLaMa for LLaMaCPP {
  type Token = llama_token;
  type Tokenizer = *const Self;
  type TokenIterator<'a> = LLaMaCPPTokenIter<'a>;

  fn tokenizer(&self) -> Self::Tokenizer {
    &*self as *const Self
  }

  fn token_iter(&mut self, params: SamplerParams) -> Self::TokenIterator<'_> {
    LLaMaCPPTokenIter { context: self, params, n_past: 0, n_last: 0 }
  }
}